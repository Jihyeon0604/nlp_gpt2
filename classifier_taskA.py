#!/usr/bin/env python3

import random, numpy as np, argparse, csv
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, get_scheduler
from sklearn.metrics import f1_score, accuracy_score

from models.gpt2 import GPT2Model
from optimizer import AdamW
from tqdm import tqdm

TQDM_DISABLE = False

def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class GPT2SentimentClassifier(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_labels = config.num_labels
        self.gpt = GPT2Model.from_pretrained()
        self.hidden_size = config.hidden_size

        for param in self.gpt.parameters():
            param.requires_grad = False
        self.classifier = torch.nn.Linear(self.hidden_size, self.num_labels)

    def unfreeze_layers(self, current_epoch):
        total_layers = len(list(self.gpt.children()))
        layers_to_unfreeze = min(current_epoch, total_layers)
        # 자식 모듈 순서대로 unfreeze
        for idx, (name, child) in enumerate(self.gpt.named_children()):
            requires_grad = idx < layers_to_unfreeze
            for param in child.parameters():
                param.requires_grad = requires_grad


    def forward(self, input_ids, attention_mask):
        outputs = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs['last_hidden_state']
        last_hidden = hidden_states[:, -1, :]
        logits = self.classifier(last_hidden)
        return logits

def load_data(filename, flag='train'):
    data, num_labels = [], {}
    with open(filename, 'r') as fp:
        for record in csv.DictReader(fp, delimiter='\t'):
            sent = record['sentence'].lower().strip()
            sent_id = record['id'].lower().strip()
            if flag == 'test':
                data.append((sent, sent_id))
            else:
                label = int(record['sentiment'].strip())
                num_labels.setdefault(label, len(num_labels))
                data.append((sent, label, sent_id))
    return (data, len(num_labels)) if flag == 'train' else data

class SentimentDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx): return self.dataset[idx]

    def pad_data(self, data):
        sents = [x[0] for x in data]
        labels = [x[1] for x in data] if len(data[0]) == 3 else None
        sent_ids = [x[-1] for x in data]
        encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
        return encoding['input_ids'], encoding['attention_mask'], labels, sents, sent_ids

    def collate_fn(self, batch):
        token_ids, attention_mask, labels, sents, sent_ids = self.pad_data(batch)
        return {
            'token_ids': token_ids,
            'attention_mask': attention_mask,
            'labels': torch.LongTensor(labels) if labels else None,
            'sents': sents,
            'sent_ids': sent_ids
        }

def model_eval(dataloader, model, device):
    model.eval()
    y_true, y_pred = [], []
    for batch in tqdm(dataloader, desc='eval', disable=TQDM_DISABLE):
        b_ids = batch['token_ids'].to(device)
        b_mask = batch['attention_mask'].to(device)
        logits = model(b_ids, b_mask).detach().cpu().numpy()
        preds = np.argmax(logits, axis=1).flatten()
        y_pred.extend(preds)
        if batch['labels'] is not None:
            y_true.extend(batch['labels'].flatten())
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average='macro')

def save_model(model, optimizer, args, config, filepath):
    torch.save({
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
    }, filepath)

def train(args):
    device = torch.device('cuda' if args.use_gpu else 'cpu')
    train_data, num_labels = load_data(args.train, 'train')
    dev_data = load_data(args.dev, 'valid')

    train_dataset = SentimentDataset(train_data, args)
    dev_dataset = SentimentDataset(dev_data, args)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=train_dataset.collate_fn)
    dev_loader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=dev_dataset.collate_fn)

    config = SimpleNamespace(hidden_dropout_prob=args.hidden_dropout_prob, num_labels=num_labels,
                             hidden_size=768, data_dir='.', fine_tune_mode=args.fine_tune_mode)
    model = GPT2SentimentClassifier(config).to(device)

    lr = args.lr
    optimizer = AdamW([
        {'params': [p for n, p in model.named_parameters() if "gpt" in n], 'lr': lr * 0.5},
        {'params': [p for n, p in model.named_parameters() if "classifier" in n], 'lr': lr}
    ])

    total_steps = len(train_loader) * args.epochs
    scheduler = get_scheduler("linear", optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    best_dev_acc = 0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        model.unfreeze_layers(epoch + 1)  # Gradual Unfreezing

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            b_ids, b_mask = batch['token_ids'].to(device), batch['attention_mask'].to(device)
            b_labels = batch['labels'].to(device)

            optimizer.zero_grad()
            logits = model(b_ids, b_mask)
            loss = F.cross_entropy(logits, b_labels.view(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        train_acc, _ = model_eval(train_loader, model, device)
        dev_acc, dev_f1 = model_eval(dev_loader, model, device)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        print(f"Epoch {epoch} | Train Loss: {train_loss / len(train_loader):.4f} | Train Acc: {train_acc:.4f} | Dev Acc: {dev_acc:.4f}")

def test(args):
    device = torch.device('cuda' if args.use_gpu else 'cpu')
    saved = torch.load(args.filepath)
    config = saved['model_config']
    model = GPT2SentimentClassifier(config)
    model.load_state_dict(saved['model'])
    model = model.to(device)

    test_data = load_data(args.test, 'test')
    test_dataset = SentimentDataset(test_data, args)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=test_dataset.collate_fn)

    test_acc, test_f1 = model_eval(test_loader, model, device)
    print(f"Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--fine_tune_mode", type=str, choices=["last-linear-layer", "full-model"], default="full-model")
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-5)  # ✔️ 낮춰서 안정화
    parser.add_argument("--train", type=str, default='data/ids-sst-train.csv')
    parser.add_argument("--dev", type=str, default='data/ids-sst-dev.csv')
    parser.add_argument("--test", type=str, default='data/ids-sst-test-student.csv')
    parser.add_argument("--filepath", type=str, default='sst-classifier-taskA.pt')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    train(args)
    test(args)
