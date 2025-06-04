#!/usr/bin/env python3

import random, numpy as np, argparse, csv
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from sklearn.metrics import f1_score, accuracy_score

from models.gpt2_lora import GPT2Model  # ✅ LoRA 적용 GPT2
from optimizer import AdamW
from tqdm import tqdm

TQDM_DISABLE = False

def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, lora_alpha=16, dropout=0.1):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.dropout = nn.Dropout(dropout)

        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.scaling = lora_alpha / r

    def forward(self, x):
        result = F.linear(x, self.weight, self.bias)
        if self.training:
            x = self.dropout(x)
        lora_result = self.lora_B(self.lora_A(x))
        return result + self.scaling * lora_result

class GPT2SentimentClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_labels = config.num_labels
        self.gpt = GPT2Model(config)  # LoRA GPT2
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = LoRALinear(config.hidden_size, self.num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.gpt(input_ids, attention_mask)
        last_token = outputs['last_token']
        dropped = self.dropout(last_token)
        logits = self.classifier(dropped)
        return logits

class SentimentDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx): return self.dataset[idx]

    def pad_data(self, data):
        sents = [x[0] for x in data]
        labels = [x[1] for x in data]
        sent_ids = [x[2] for x in data]
        encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
        return torch.LongTensor(encoding['input_ids']), torch.LongTensor(encoding['attention_mask']), torch.LongTensor(labels), sents, sent_ids

    def collate_fn(self, batch):
        token_ids, attention_mask, labels, sents, sent_ids = self.pad_data(batch)
        return {'token_ids': token_ids, 'attention_mask': attention_mask, 'labels': labels, 'sents': sents, 'sent_ids': sent_ids}

class SentimentTestDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx): return self.dataset[idx]

    def pad_data(self, data):
        sents = [x[0] for x in data]
        sent_ids = [x[1] for x in data]
        encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
        return torch.LongTensor(encoding['input_ids']), torch.LongTensor(encoding['attention_mask']), sents, sent_ids

    def collate_fn(self, batch):
        token_ids, attention_mask, sents, sent_ids = self.pad_data(batch)
        return {'token_ids': token_ids, 'attention_mask': attention_mask, 'sents': sents, 'sent_ids': sent_ids}

def load_data(filename, flag='train'):
    data, label_dict = [], {}
    with open(filename, 'r') as fp:
        for record in csv.DictReader(fp, delimiter='\t'):
            sent = record['sentence'].lower().strip()
            sent_id = record['id'].lower().strip()
            if flag == 'test':
                data.append((sent, sent_id))
            else:
                label = int(record['sentiment'].strip())
                if label not in label_dict:
                    label_dict[label] = len(label_dict)
                data.append((sent, label, sent_id))
    print(f"load {len(data)} data from {filename}") if flag != 'test' else None
    return (data, len(label_dict)) if flag != 'test' else data

def model_eval(dataloader, model, device):
    model.eval()
    y_true, y_pred, sents, sent_ids = [], [], [], []
    for batch in tqdm(dataloader, desc='eval', disable=TQDM_DISABLE):
        b_ids, b_mask = batch['token_ids'].to(device), batch['attention_mask'].to(device)
        logits = model(b_ids, b_mask).detach().cpu().numpy()
        preds = np.argmax(logits, axis=1).flatten()
        y_true.extend(batch['labels'].flatten())
        y_pred.extend(preds)
        sents.extend(batch['sents'])
        sent_ids.extend(batch['sent_ids'])
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average='macro'), y_pred, y_true, sents, sent_ids

def model_test_eval(dataloader, model, device):
    model.eval()
    y_pred, sents, sent_ids = [], [], []
    for batch in tqdm(dataloader, desc='eval', disable=TQDM_DISABLE):
        b_ids, b_mask = batch['token_ids'].to(device), batch['attention_mask'].to(device)
        logits = model(b_ids, b_mask).detach().cpu().numpy()
        preds = np.argmax(logits, axis=1).flatten()
        y_pred.extend(preds)
        sents.extend(batch['sents'])
        sent_ids.extend(batch['sent_ids'])
    return y_pred, sents, sent_ids

def save_model(model, optimizer, args, config, filepath):
    torch.save({
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state()
    }, filepath)
    print(f"save the model to {filepath}")

def train(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    train_data, num_labels = load_data(args.train, 'train')
    dev_data = load_data(args.dev, 'valid')
    train_loader = DataLoader(SentimentDataset(train_data, args), batch_size=args.batch_size, shuffle=True, collate_fn=SentimentDataset(train_data, args).collate_fn)
    dev_loader = DataLoader(SentimentDataset(dev_data, args), batch_size=args.batch_size, shuffle=False, collate_fn=SentimentDataset(dev_data, args).collate_fn)
    
    config_args = vars(args).copy()
    config_args.update({
    'num_labels': num_labels,
    'hidden_size': 768,
    'name_or_path': "gpt2",
    'vocab_size': 50257,
    'pad_token_id': 50256,
    'max_position_embeddings': 1024,
    'num_hidden_layers': 12,
    })
    config = SimpleNamespace(**config_args)

    model = GPT2SentimentClassifier(config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    best_dev_acc = 0
    for epoch in range(args.epochs):
        model.train()
        train_loss, num_batches = 0, 0
        for batch in tqdm(train_loader, desc=f"train-{epoch}", disable=TQDM_DISABLE):
            b_ids, b_mask, b_labels = batch['token_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
            optimizer.zero_grad()
            logits = model(b_ids, b_mask)
            loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            num_batches += 1
        train_acc, train_f1, *_ = model_eval(train_loader, model, device)
        dev_acc, dev_f1, *_ = model_eval(dev_loader, model, device)
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)
        print(f"Epoch {epoch}: train loss :: {train_loss / num_batches:.3f}, train acc :: {train_acc:.3f}, train F1 :: {train_f1:.3f}, dev acc :: {dev_acc:.3f}, dev F1 :: {dev_f1:.3f}")

def test(args):
    from types import SimpleNamespace
    import torch.serialization
    import numpy as np
    torch.serialization.add_safe_globals([SimpleNamespace, np.ndarray])
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    saved = torch.load(args.filepath, weights_only=False)
    model = GPT2SentimentClassifier(saved['model_config'])
    model.load_state_dict(saved['model'])
    model = model.to(device)

    dev_loader = DataLoader(SentimentDataset(load_data(args.dev, 'valid')[0], args), batch_size=args.batch_size, collate_fn=SentimentDataset(load_data(args.dev, 'valid')[0], args).collate_fn)
    test_loader = DataLoader(SentimentTestDataset(load_data(args.test, 'test'), args), batch_size=args.batch_size, collate_fn=SentimentTestDataset(load_data(args.test, 'test'), args).collate_fn)

    dev_acc, dev_f1, dev_pred, dev_true, dev_sents, dev_sent_ids = model_eval(dev_loader, model, device)
    print("DONE DEV")
    test_pred, test_sents, test_sent_ids = model_test_eval(test_loader, model, device)
    print("DONE TEST")

    with open(args.dev_out, 'w+') as f:
        f.write("id,Predicted_Sentiment\n")
        for pid, pred in zip(dev_sent_ids, dev_pred):
            f.write(f"{pid},{pred}\n")
    with open(args.test_out, 'w+') as f:
        f.write("id,Predicted_Sentiment\n")
        for pid, pred in zip(test_sent_ids, test_pred):
            f.write(f"{pid},{pred}\n")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine-tune-mode", type=str, choices=['full-model'], default="full-model")
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    sst_batch_size = args.batch_size if args.batch_size > 0 else 64
    args.batch_size = sst_batch_size  

    sst_args = SimpleNamespace(**{**vars(args), 
    'filepath': 'sst-classifier.pt',
    'train': 'data/ids-sst-train.csv',
    'dev': 'data/ids-sst-dev.csv',
    'test': 'data/ids-sst-test-student.csv',
    'dev_out': 'predictions/full-model-sst-dev-out.csv',
    'test_out': 'predictions/full-model-sst-test-out.csv',
    'batch_size': 64,
    'name_or_path': 'gpt2',
    'vocab_size': tokenizer.vocab_size,       
    'pad_token_id': tokenizer.pad_token_id,
    'max_position_embeddings': 1024
})


    print("Training Sentiment Classifier on SST...")
    train(sst_args)
    print("Evaluating on SST...")
    test(sst_args)
    
    ids_batch_size = args.batch_size if args.batch_size > 0 else 64
    args.batch_size = ids_batch_size 

    cfimdb_args = SimpleNamespace(**{
    **vars(args),
    'filepath': 'cfimdb-classifier.pt',
    'train': 'data/ids-cfimdb-train.csv',
    'dev': 'data/ids-cfimdb-dev.csv',
    'test': 'data/ids-cfimdb-test-student.csv',
    'dev_out': 'predictions/full-model-cfimdb-dev-out.csv',
    'test_out': 'predictions/full-model-cfimdb-test-out.csv',
    'batch_size': 8,
    'name_or_path': 'gpt2',
    'vocab_size': tokenizer.vocab_size,       
    'pad_token_id': tokenizer.pad_token_id,
    'max_position_embeddings': 1024
})

    print("Training Sentiment Classifier on CFIMDB...")
    train(cfimdb_args)
    print("Evaluating on CFIMDB...")
    test(cfimdb_args)
