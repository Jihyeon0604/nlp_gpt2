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

    # Unlikelihood Loss 관련 파라미터
    self.alpha = args.alpha if hasattr(args, 'alpha') else 1.0  # Unlikelihood loss weight
    self.repetition_window = args.repetition_window if hasattr(args, 'repetition_window') else 5
    
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

  def compute_unlikelihood_loss(self, logits, input_ids):
    """Unlikelihood Loss 계산 함수"""
    batch_size, seq_len, vocab_size = logits.shape
    total_ul_loss = 0.0
    
    for b in range(batch_size):
      for t in range(1, seq_len):  # 첫 번째 토큰은 스킵
        current_token = input_ids[b, t]
        
        # 현재 위치에서 과거 repetition_window 내의 토큰들 확인
        start_idx = max(0, t - self.repetition_window)
        context_tokens = input_ids[b, start_idx:t]
        
        # 반복되는 토큰들 찾기
        repeated_tokens = []
        for prev_token in context_tokens:
          if prev_token == current_token:
            repeated_tokens.append(prev_token.item())
        
        # 반복된 토큰들에 대해 unlikelihood loss 적용
        if repeated_tokens:
          current_logits = logits[b, t]  # [vocab_size]
          probs = torch.softmax(current_logits, dim=-1)
          
          # 반복된 토큰들의 확률을 낮추는 손실
          for token_id in set(repeated_tokens):  # 중복 제거
            if token_id < vocab_size:
              # -log(1 - p(token)) : 확률이 높을수록 손실이 커짐
              token_prob = probs[token_id]
              ul_loss = -torch.log(torch.clamp(1.0 - token_prob, min=1e-8))
              total_ul_loss += ul_loss
    
    return total_ul_loss / (batch_size * seq_len)  # 정규화

  def compute_combined_loss(self, logits, labels, input_ids):
    """Standard Cross-Entropy + Unlikelihood Loss"""
    # 기본 언어 모델링 손실
    ce_logits = rearrange(logits[:, :-1], 'b t d -> (b t) d')
    ce_labels = labels[:, 1:].contiguous().flatten()
    ce_loss = F.cross_entropy(ce_logits, ce_labels, reduction='mean')
    
    # Unlikelihood 손실
    ul_loss = self.compute_unlikelihood_loss(logits, input_ids)
    
    # 결합된 손실
    total_loss = ce_loss + self.alpha * ul_loss
    
    return total_loss, ce_loss, ul_loss

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
    ce_loss_sum = 0
    ul_loss_sum = 0
    num_batches = 0

    for batch in tqdm(sonnet_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
        b_ids, b_mask = batch['token_ids'].to(device), batch['attention_mask'].to(device)
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(b_ids, b_mask)
        
        # Unlikelihood Loss와 함께 손실 계산
        total_loss, ce_loss, ul_loss = model.compute_combined_loss(logits, b_ids, b_ids)
        
        # Backward pass
        total_loss.backward()
        optimizer.step()

        # 통계 수집
        train_loss += total_loss.item()
        ce_loss_sum += ce_loss.item()
        ul_loss_sum += ul_loss.item()
        num_batches += 1

    # 에포크 통계
    avg_train_loss = train_loss / num_batches
    avg_ce_loss = ce_loss_sum / num_batches
    avg_ul_loss = ul_loss_sum / num_batches
    
    scheduler.step(avg_train_loss)
    print(f"Epoch {epoch}: total loss :: {avg_train_loss:.3f} | CE loss :: {avg_ce_loss:.3f} | UL loss :: {avg_ul_loss:.3f}")

    if avg_train_loss < best_loss:
        best_loss = avg_train_loss
        counter = 0
        print(f"Saving best model at epoch {epoch} with total loss {avg_train_loss:.3f}")
        save_model(model, optimizer, args, f'best_{args.filepath}')
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            break

    # 매 에포크마다 간단한 샘플 생성 (선택사항)
    if epoch % 2 == 0:  # 2 에포크마다만 샘플 생성
        print('Generating sample output sonnets...')
        model.eval()
        sample_count = 0
        for batch in held_out_sonnet_dataset:
            if sample_count >= 2:  # 최대 2개만 샘플로 생성
                break
            encoding = model.tokenizer(batch[1], return_tensors='pt', padding=True, truncation=True)
            encoding = {k: v.to(device) for k, v in encoding.items()}
            output = model.generate(encoding['input_ids'], temperature=args.temperature)
            print(f'Sample {sample_count + 1}:')
            print(f'{batch[1]}{output[1]}\n')
            sample_count += 1
        
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
  parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets_A.txt")
  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')
  parser.add_argument("--temperature", type=float, default=0.8)
  parser.add_argument("--top_p", type=float, default=0.9) 
  parser.add_argument("--batch_size", type=int, default=8)
  parser.add_argument("--lr", type=float, default=1e-5)
  parser.add_argument("--model_size", type=str, choices=['gpt2', 'gpt2-medium', 'gpt2-large'], default='gpt2')
  
  # Unlikelihood Loss 관련 파라미터
  parser.add_argument("--alpha", type=float, default=1.0, help="Weight for unlikelihood loss")
  parser.add_argument("--repetition_window", type=int, default=5, help="Window size for detecting repetitions")
  
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
  args.filepath = f'{args.epochs}-{args.lr}-sonnet_A.pt'
  seed_everything(args.seed)
  
  # 훈련 실행
  trained_model = train(args)
  
  # 훈련 완료 후 최종 소네트 생성
  print("\n" + "="*50)
  print("Training completed! Generating final submission sonnets...")
  print("="*50)
  generate_submission_sonnets(args)