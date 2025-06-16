 
'''
Paraphrase detection을 위한 시작 코드.

고려 사항:
 - ParaphraseGPT: 여러분이 구현한 GPT-2 분류 모델 .
 - train: Quora paraphrase detection 데이터셋에서 ParaphraseGPT를 훈련시키는 절차.
 - test: Test 절차. 프로젝트 결과 제출에 필요한 파일들을 생성함.

실행:
  `python paraphrase_detection.py --use_gpu`
ParaphraseGPT model을 훈련 및 평가하고, 필요한 제출용 파일을 작성한다.
'''
import argparse
import random
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score

from datasets import (
  ParaphraseDetectionDataset,
  ParaphraseDetectionTestDataset,
  load_paraphrase_data
)
from evaluation import model_eval_paraphrase, model_test_paraphrase
from models.gpt2 import GPT2Model
from optimizer import AdamW
from torch.serialization import safe_globals
from argparse import Namespace
import numpy.core

TQDM_DISABLE = False

def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True

class ParaphraseGPT(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.paraphrase_detection_head = nn.Linear(args.d, 2)
    for param in self.gpt.parameters():
      param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    outputs = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
    sequence_output = outputs['last_hidden_state']
    last_token_indices = attention_mask.sum(dim=1) - 1
    last_hidden = sequence_output[torch.arange(sequence_output.size(0)), last_token_indices]
    logits = self.paraphrase_detection_head(last_hidden)
    return logits

def save_model(model, optimizer, args, filepath):
  torch.save({
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }, filepath)
  print(f"Model saved to {filepath}")

def train(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  para_train_data = ParaphraseDetectionDataset(load_paraphrase_data(args.para_train), args)
  para_dev_data = ParaphraseDetectionDataset(load_paraphrase_data(args.para_dev), args)
  train_loader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size, collate_fn=para_train_data.collate_fn)
  dev_loader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size, collate_fn=para_dev_data.collate_fn)

  args = add_arguments(args)
  model = ParaphraseGPT(args).to(device)
  optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.)
  best_acc = 0

  for epoch in range(args.epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"[Train Epoch {epoch}]"):
      b_ids, b_mask, labels = batch['token_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].flatten().to(device)
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      loss = F.cross_entropy(logits, labels)
      loss.backward()
      optimizer.step()
      total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    acc, f1, preds, trues, _ = model_eval_paraphrase(dev_loader, model, device)
    precision = precision_score(trues, preds)
    recall = recall_score(trues, preds)

    if acc > best_acc:
      best_acc = acc
      save_model(model, optimizer, args, args.filepath)

    print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

@torch.no_grad()
def test(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  with safe_globals([Namespace, np.core.multiarray._reconstruct]):
    saved = torch.load(args.filepath, weights_only=False)

  model = ParaphraseGPT(saved['args']).to(device)
  model.load_state_dict(saved['model'])
  model.eval()
  print(f"Loaded model from {args.filepath}")

  dev_data = ParaphraseDetectionDataset(load_paraphrase_data(args.para_dev), args)
  test_data = ParaphraseDetectionTestDataset(load_paraphrase_data(args.para_test, split='test'), args)
  dev_loader = DataLoader(dev_data, batch_size=args.batch_size, shuffle=False, collate_fn=dev_data.collate_fn)
  test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=test_data.collate_fn)

  acc, f1, preds, trues, sent_ids = model_eval_paraphrase(dev_loader, model, device)
  precision = precision_score(trues, preds)
  recall = recall_score(trues, preds)
  print(f"Dev Set: Acc: {acc:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

  test_preds, test_ids = model_test_paraphrase(test_loader, model, device)
  with open(args.para_dev_out, "w") as f:
    f.write("id\tPredicted_Is_Paraphrase\n")
    for sid, pred in zip(sent_ids, preds):
      f.write(f"{sid},{pred}\n")
  with open(args.para_test_out, "w") as f:
    f.write("id\tPredicted_Is_Paraphrase\n")
    for sid, pred in zip(test_ids, test_preds):
      f.write(f"{sid},{pred}\n")

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
  parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
  parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")
  parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
  parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")
  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action="store_true")
  parser.add_argument("--batch_size", type=int, default=8)
  parser.add_argument("--lr", type=float, default=1e-5)
  parser.add_argument("--model_size", type=str, choices=['gpt2', 'gpt2-medium', 'gpt2-large'], default='gpt2')
  return parser.parse_args()

def add_arguments(args):
  if args.model_size == 'gpt2':
    args.d, args.l, args.num_heads = 768, 12, 12
  elif args.model_size == 'gpt2-medium':
    args.d, args.l, args.num_heads = 1024, 24, 16
  elif args.model_size == 'gpt2-large':
    args.d, args.l, args.num_heads = 1280, 36, 20
  else:
    raise ValueError("Unsupported model size")
  return args

if __name__ == "__main__":
  args = get_args()
  args.filepath = f'{args.epochs}-{args.lr}-paraphrase.pt'
  seed_everything(args.seed)
  train(args)
  test(args)
