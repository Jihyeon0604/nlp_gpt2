import argparse
import random
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer
from einops import rearrange

from datasets import (
  SonnetsDataset,
)
from models.gpt2 import GPT2Model

from optimizer import AdamW

TQDM_DISABLE = False

# 재현성을 위한 random seed 고정.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


class SonnetGPT(nn.Module):
  """Sonnet 생성을 위해 설계된 여러분의 GPT-2 모델."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

    # 마지막 일부 레이어만 fine-tuning (예: 마지막 4개 레이어)
    for param in self.gpt.parameters():
      param.requires_grad = True

    # 일부 레이어만 fine-tuning (마지막 4개만)
    num_layers = len(self.gpt.gpt_layers)
    for i, block in enumerate(self.gpt.gpt_layers):
      for param in block.parameters():
        param.requires_grad = i >= (num_layers - 4)

  def forward(self, input_ids, attention_mask):
    outputs = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
    hidden_states = outputs["last_hidden_state"]  # shape: [batch_size, seq_len, hidden_dim]

    # hidden states를 vocabulary logit으로 변환
    logits = self.gpt.hidden_state_to_token(hidden_states)  # shape: [batch_size, seq_len, vocab_size]
    return logits

  def get_device(self):
    for param in self.gpt.parameters():
      return param.device

  @torch.no_grad()
  def generate(self, encoding, temperature=0.7, top_p=0.9, max_length=128, target_lines=14):
    """Top-p sampling 기반 14줄 소네트 생성 함수 (문장 길이 제어 버전)"""
    input_ids = encoding.to(self.get_device())
    generated = input_ids.clone()
    
    # 구두점 토큰 ID들 (문장 종료를 위한)
    punctuation_tokens = {
        self.tokenizer.encode('.')[0]: 2.0,    # 마침표
        self.tokenizer.encode(',')[0]: 1.5,    # 쉼표  
        self.tokenizer.encode(';')[0]: 1.5,    # 세미콜론
        self.tokenizer.encode(':')[0]: 1.3,    # 콜론
        self.tokenizer.encode('!')[0]: 1.8,    # 느낌표
        self.tokenizer.encode('?')[0]: 1.8,    # 물음표
    }

    for step in range(max_length):
        # 현재 생성된 텍스트 디코딩
        decoded_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        current_lines = decoded_text.count('\n')

        # 🎯 14줄 이상이면 중단
        if current_lines >= target_lines:
            break

        # 모델 forward
        attention_mask = torch.ones_like(generated).to(self.get_device())
        logits = self.forward(generated, attention_mask)[:, -1, :]  # [B, vocab]

        # 문장 길이 제어: 현재 줄의 단어 수 계산
        current_line_text = decoded_text.split('\n')[-1] if '\n' in decoded_text else decoded_text
        current_line_words = len(current_line_text.split())
        
        # 문장이 너무 길어지면 구두점 생성을 장려
        if current_line_words > 8:  # 8단어 이상이면 구두점 장려
            punctuation_boost = min(2.0, (current_line_words - 8) * 0.3)  # 최대 2배까지
            for token_id, base_boost in punctuation_tokens.items():
                if token_id < logits.size(-1):
                    logits[0, token_id] += punctuation_boost * base_boost
        
        # 매우 긴 문장의 경우 줄바꿈 강제 장려
        if current_line_words > 12:
            newline_token = self.tokenizer.encode('\n')[0] if self.tokenizer.encode('\n') else None
            if newline_token and newline_token < logits.size(-1):
                logits[0, newline_token] += 3.0

        # Temperature 적용 및 numerical stability 보장
        logits = logits / max(temperature, 1e-8)
        
        # Top-p 샘플링 (수정된 버전)
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = probs.cumsum(dim=-1)

        # Top-p 필터링
        sorted_mask = cumulative_probs > top_p
        # 최소한 하나의 토큰은 유지하도록 보장
        if sorted_mask.all():
            sorted_mask[0] = False
        
        # 마스크된 위치를 매우 작은 값으로 설정 (완전히 -inf로 하지 않음)
        sorted_logits = sorted_logits.clone()
        sorted_logits[sorted_mask] = sorted_logits[sorted_mask] - 1e10
        
        # 다시 softmax 적용
        probs = torch.softmax(sorted_logits, dim=-1)
        
        # Numerical stability 체크
        if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
            # 문제가 있으면 uniform distribution으로 fallback
            probs = torch.ones_like(probs) / probs.size(-1)
        
        # 확률 정규화 (혹시 모를 문제 방지)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        
        # 샘플링
        try:
            next_token = torch.multinomial(probs, num_samples=1)
            next_token_id = sorted_indices.gather(-1, next_token)  # shape: [1, 1]
        except RuntimeError:
            # multinomial이 실패하면 가장 높은 확률의 토큰 선택
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
            next_token_id = sorted_indices.gather(-1, next_token)

        # 다음 토큰 붙이기
        generated = torch.cat([generated, next_token_id], dim=1)

    # 디코딩 후 정확히 14줄로 제한
    decoded_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
    lines = decoded_text.strip().split('\n')
    trimmed_text = '\n'.join(lines[:target_lines])

    return generated, trimmed_text


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
  sonnet_dataset = SonnetsDataset(args.sonnet_path)
  sonnet_dataloader = DataLoader(sonnet_dataset, shuffle=True, batch_size=args.batch_size,
                                 collate_fn=sonnet_dataset.collate_fn)
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  args = add_arguments(args)
  model = SonnetGPT(args).to(device)
  optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

  best_loss = float('inf')
  patience = 3
  counter = 0

  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0

    for batch in tqdm(sonnet_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
        b_ids, b_mask = batch['token_ids'].to(device), batch['attention_mask'].to(device)
        optimizer.zero_grad()
        logits = model(b_ids, b_mask)
        logits = rearrange(logits[:, :-1], 'b t d -> (b t) d')
        labels = b_ids[:, 1:].contiguous().flatten()
        loss = F.cross_entropy(logits, labels, reduction='mean')
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        num_batches += 1

    train_loss = train_loss / num_batches
    scheduler.step(train_loss)
    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}.")

    if train_loss < best_loss:
        best_loss = train_loss
        counter = 0
        print(f"Saving best model at epoch {epoch} with loss {train_loss:.3f}")
        save_model(model, optimizer, args, f'best_{args.filepath}')
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            break
        
  print("Training completed!")
  return model

@torch.no_grad()
def generate_submission_sonnets(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  # 최종 훈련된 모델 또는 best 모델 로드
  try:
    saved = torch.load(f'best_{args.filepath}', weights_only=False)
    print("Loading best model for generation...")
  except FileNotFoundError:
    saved = torch.load(f'{args.epochs-1}_{args.filepath}', weights_only=False)
    print("Loading final epoch model for generation...")
  
  model = SonnetGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device).eval()

  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)
  generated_sonnets = []

  print("Generating submission sonnets...")
  for batch in held_out_sonnet_dataset:
    sonnet_id = batch[0]
    encoding = model.tokenizer(batch[1], return_tensors='pt', padding=False, truncation=True)
    encoding = {k: v.to(device) for k, v in encoding.items()}
    output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)
    decoded_output = model.tokenizer.decode(output[0][0])
    generated_sonnets.append((sonnet_id, f'{decoded_output}\n\n'))

  with open(args.sonnet_out, "w+") as f:
    f.write(f"--Generated Sonnets-- \n\n")
    for sonnet in generated_sonnets:
      f.write(f"\n{sonnet[0]}\n")
      f.write(sonnet[1])
  
  print(f"Generated sonnets saved to {args.sonnet_out}")


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
  parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out.txt")
  parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets.txt")
  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')
  parser.add_argument("--temperature", type=float, default=0.8)
  parser.add_argument("--top_p", type=float, default=0.9)
  parser.add_argument("--batch_size", type=int, default=8)
  parser.add_argument("--lr", type=float, default=1e-5)
  parser.add_argument("--model_size", type=str, choices=['gpt2', 'gpt2-medium', 'gpt2-large'], default='gpt2')
  return parser.parse_args()

def add_arguments(args):
  if args.model_size == 'gpt2':
    args.d = 768
    args.l = 12
    args.num_heads = 12
  elif args.model_size == 'gpt2-medium':
    args.d = 1024
    args.l = 24
    args.num_heads = 16
  elif args.model_size == 'gpt2-large':
    args.d = 1280
    args.l = 36
    args.num_heads = 20
  else:
    raise Exception(f'{args.model_size} is not supported.')
  return args

if __name__ == "__main__":
  args = get_args()
  args.filepath = f'{args.epochs}-{args.lr}-sonnet.pt'
  seed_everything(args.seed)
  
  # 훈련 실행
  trained_model = train(args)
  
  # 훈련 완료 후 최종 소네트 생성
  print("\n" + "="*50)
  print("Training completed! Generating final submission sonnets...")
  print("="*50)
  generate_submission_sonnets(args)