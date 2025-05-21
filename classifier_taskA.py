#!/usr/bin/env python3

import random, numpy as np, argparse, csv
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
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

class GPT2SentimentClassifier(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_labels = config.num_labels
        self.gpt = GPT2Model.from_pretrained()
        self.hidden_size = config.hidden_size
        self.classifier = torch.nn.Linear(self.hidden_size, self.num_labels)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.fine_tune_mode = config.fine_tune_mode
        self.freeze_layers()

    def freeze_layers(self):
        for param in self.gpt.parameters():
            param.requires_grad = (self.fine_tune_mode == 'full-model')
        if self.fine_tune_mode == 'last-linear-layer':
            for param in self.gpt.parameters():
                param.requires_grad = False

    def gradual_unfreeze(self, current_epoch, total_epochs):
        layers = self.gpt.gpt_layers
        num_layers = len(layers)
        layers_to_unfreeze = (current_epoch + 1) * num_layers // total_epochs
        for i in range(layers_to_unfreeze):
            for param in layers[i].parameters():
                param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        outputs = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
        last_token = outputs['last_token']
        pooled_output = self.dropout(last_token)
        logits = self.classifier(pooled_output)
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
                if label not in num_labels:
                    num_labels[label] = len(num_labels)
                data.append((sent, label, sent_id))
    return (data, len(num_labels)) if flag == 'train' else data

class SentimentDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sents = [x[0] for x in data]
        labels = [x[1] for x in data] if len(data[0]) == 3 else None
        encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
        return encoding['input_ids'], encoding['attention_mask'], labels

    def collate_fn(self, batch):
        token_ids, attention_mask, labels = self.pad_data(batch)
        return {
            'token_ids': token_ids,
            'attention_mask': attention_mask,
            'labels': torch.LongTensor(labels) if labels is not None else None
        }

def model_eval(dataloader, model, device):
    model.eval()
    y_true, y_pred = [], []
    for batch in dataloader:
        b_ids = batch['token_ids'].to(device)
        b_mask = batch['attention_mask'].to(device)
        logits = model(b_ids, b_mask).detach().cpu().numpy()
        preds = np.argmax(logits, axis=1).flatten()
        y_pred.extend(preds)
        if batch['labels'] is not None:
            y_true.extend(batch['labels'].flatten())
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average='macro')

def slanted_triangular_lr(step, total_steps, max_lr, cut_frac=0.1, ratio=32):
    cut = int(total_steps * cut_frac)
    p = step / cut if step < cut else 1 - (step - cut) / (total_steps - cut)
    return max_lr * (1 + p * (ratio - 1)) / ratio

def build_discriminative_optimizer(model, lr):
    layers = model.gpt.gpt_layers
    grouped = []
    for i, layer in enumerate(layers):
        grouped.append({'params': layer.parameters(), 'lr': lr / (2.6 ** (11 - i))})
    grouped.append({'params': model.classifier.parameters(), 'lr': lr})
    return AdamW(grouped)

def train(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    train_data, num_labels = load_data(args.train, 'train')
    dev_data = load_data(args.dev, 'valid')

    train_dataset = SentimentDataset(train_data)
    dev_dataset = SentimentDataset(dev_data)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=train_dataset.collate_fn)
    dev_loader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=dev_dataset.collate_fn)

    config = SimpleNamespace(
        hidden_dropout_prob=args.hidden_dropout_prob,
        num_labels=num_labels,
        hidden_size=768,
        fine_tune_mode=args.fine_tune_mode
    )

    model = GPT2SentimentClassifier(config).to(device)
    total_steps = len(train_loader) * args.epochs
    best_dev_acc = 0
    global_step = 0
    optimizer = build_discriminative_optimizer(model, args.lr)

    for epoch in range(args.epochs):
        model.train()
        model.gradual_unfreeze(epoch, args.epochs)
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            b_ids = batch['token_ids'].to(device)
            b_mask = batch['attention_mask'].to(device)
            b_labels = batch['labels'].to(device)

            lr = slanted_triangular_lr(global_step, total_steps, args.lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            optimizer.zero_grad()
            logits = model(b_ids, b_mask)
            loss = F.cross_entropy(logits, b_labels.view(-1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            global_step += 1

        train_acc, train_f1 = model_eval(train_loader, model, device)
        dev_acc, dev_f1 = model_eval(dev_loader, model, device)
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            torch.save(model.state_dict(), args.filepath)

        print(f"Epoch {epoch} | Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f} | Dev Acc: {dev_acc:.4f} | Dev F1: {dev_f1:.4f}")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--fine_tune_mode", type=str, choices=["last-linear-layer", "full-model"], default="last-linear-layer")
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--dev", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--filepath", type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    train(args)
