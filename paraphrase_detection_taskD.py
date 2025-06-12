import argparse
import random
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from datasets import (
  ParaphraseDetectionDataset,
  ParaphraseDetectionTestDataset,
  load_paraphrase_data
)
from evaluation import model_eval_paraphrase, model_test_paraphrase
from models.gpt2 import GPT2Model
from optimizer import AdamW

# safe_globals로 Namespace 허용
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
  """Paraphrase Detection을 위해 설계된 여러분의 GPT-2 Model."""
  # 1. ParaphraseGPT 클래스에 Dropout 추가

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.dropout = nn.Dropout(p=0.3)  # Dropout 추가
    self.paraphrase_detection_head = nn.Linear(args.d, 2)  # yes or no

    # 전체 파라미터를 파인튜닝하도록 설정
    for param in self.gpt.parameters():
      param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    outputs = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
    sequence_output = outputs['last_hidden_state']

    # Mean Pooling
    mask = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
    sum_hidden = (sequence_output * mask).sum(dim=1)
    mean_hidden = sum_hidden / mask.sum(dim=1)

    # Multi-Sample Dropout (3회)
    logits = 0
    for _ in range(3):
      logits += self.paraphrase_detection_head(self.dropout(mean_hidden))
    logits /= 3
    return logits

# 2. 추가: Label Smoothing Loss 함수
def smooth_cross_entropy(logits, labels, smoothing=0.1):
  confidence = 1.0 - smoothing
  logprobs = F.log_softmax(logits, dim=-1)
  nll = -logprobs.gather(dim=-1, index=labels.unsqueeze(1)).squeeze(1)
  smooth_loss = -logprobs.mean(dim=-1)
  return (confidence * nll + smoothing * smooth_loss).mean()

def save_model(model, optimizer, args, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }
  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")


def train(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  para_train_data = load_paraphrase_data(args.para_train)
  para_dev_data = load_paraphrase_data(args.para_dev)

  para_train_data = ParaphraseDetectionDataset(para_train_data, args)
  para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)

  para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                     collate_fn=para_train_data.collate_fn)
  para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                   collate_fn=para_dev_data.collate_fn)

  args = add_arguments(args)
  model = ParaphraseGPT(args).to(device)
  # optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.)
  # 3. LLRD Optimizer 적용
  optimizer = AdamW([
    {"params": model.gpt.gpt_layers[:args.l//2].parameters(), "lr": args.lr * 0.5},
    {"params": model.gpt.gpt_layers[args.l//2:].parameters(), "lr": args.lr},
    {"params": model.paraphrase_detection_head.parameters(), "lr": args.lr * 2}
  ], weight_decay=0.)
  
  # CosineAnnealingLR scheduler
  scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
  
  best_dev_acc = 0

  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0

    for batch in tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      b_ids, b_mask, labels = batch['token_ids'], batch['attention_mask'], batch['labels'].flatten()
      b_ids, b_mask, labels = b_ids.to(device), b_mask.to(device), labels.to(device)

      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      # loss = F.cross_entropy(logits, labels, reduction='mean')
      loss = smooth_cross_entropy(logits, labels, smoothing=0.1)  # Label Smoothing Loss 적용
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1

    scheduler.step()  # update learning rate
    train_loss = train_loss / num_batches
    dev_acc, dev_f1, *_ = model_eval_paraphrase(para_dev_dataloader, model, device)

    if dev_acc > best_dev_acc:
      best_dev_acc = dev_acc
      save_model(model, optimizer, args, args.filepath)

    print(f"Epoch {epoch}: train loss :: {train_loss:.3f}, dev acc :: {dev_acc:.3f}")


@torch.no_grad()
def test(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  # 안전한 global 등록 + weights_only=False 설정
  with safe_globals([Namespace, np.core.multiarray._reconstruct]):
    saved = torch.load(args.filepath, weights_only=False)

  model = ParaphraseGPT(saved['args']).to(device)
  model.load_state_dict(saved['model'])
  model.eval()
  print(f"Loaded model to test from {args.filepath}")
  
  para_dev_data = load_paraphrase_data(args.para_dev)
  para_test_data = load_paraphrase_data(args.para_test, split='test')

  para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)
  para_test_data = ParaphraseDetectionTestDataset(para_test_data, args)

  para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                   collate_fn=para_dev_data.collate_fn)
  para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                    collate_fn=para_test_data.collate_fn)

  dev_para_acc, _, dev_para_y_pred, _, dev_para_sent_ids = model_eval_paraphrase(para_dev_dataloader, model, device)
  print(f"dev paraphrase acc :: {dev_para_acc :.3f}")
  test_para_y_pred, test_para_sent_ids = model_test_paraphrase(para_test_dataloader, model, device)

  with open(args.para_dev_out, "w+") as f:
    f.write(f"id \t Predicted_Is_Paraphrase \n")
    for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
      f.write(f"{p}, {s} \n")

  with open(args.para_test_out, "w+") as f:
    f.write(f"id \t Predicted_Is_Paraphrase \n")
    for p, s in zip(test_para_sent_ids, test_para_y_pred):
      f.write(f"{p}, {s} \n")


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
  parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
  parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")
  parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
  parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")
  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')
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
    raise Exception(f'{args.model_size} is not supported.')
  return args


if __name__ == "__main__":
  args = get_args()
  args.filepath = f'{args.epochs}-{args.lr}-paraphrase_taskD.pt'
  seed_everything(args.seed)
  train(args)
  test(args)