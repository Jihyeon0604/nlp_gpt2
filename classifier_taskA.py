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
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class GPT2SentimentClassifier(torch.nn.Module):
    def __init__(self, config):
        super(GPT2SentimentClassifier, self).__init__()
        self.num_labels = config.num_labels
        self.gpt = GPT2Model.from_pretrained()
        self.hidden_size = config.hidden_size

        assert config.fine_tune_mode in ["last-linear-layer", "full-model"]
        for param in self.gpt.parameters():
            param.requires_grad = (config.fine_tune_mode == 'full-model')

        self.classifier = torch.nn.Linear(self.hidden_size, self.num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        last_hidden = hidden_states[:, -1, :]
        logits = self.classifier(last_hidden)
        return logits

def load_data(filename, flag='train'):
    num_labels = {}
    data = []
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
    if flag == 'train':
        return data, len(num_labels)
    return data

class SentimentDataset(Dataset):
    def __init__(self, dataset, args):
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
        sent_ids = [x[-1] for x in data]

        encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
        token_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        return token_ids, attention_mask, labels, sents, sent_ids

    def collate_fn(self, all_data):
        token_ids, attention_mask, labels, sents, sent_ids = self.pad_data(all_data)
        return {
            'token_ids': token_ids,
            'attention_mask': attention_mask,
            'labels': torch.LongTensor(labels) if labels is not None else None,
            'sents': sents,
            'sent_ids': sent_ids
        }

def model_eval(dataloader, model, device):
    model.eval()
    y_true, y_pred = [], []
    for batch in tqdm(dataloader, desc='eval', disable=TQDM_DISABLE):
        b_ids = batch['token_ids'].to(device)
        b_mask = batch['attention_mask'].to(device)
        b_labels = batch['labels']
        logits = model(b_ids, b_mask).detach().cpu().numpy()
        preds = np.argmax(logits, axis=1).flatten()
        y_pred.extend(preds)
        if b_labels is not None:
            y_true.extend(b_labels.flatten())
    f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)
    return acc, f1

def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }
    torch.save(save_info, filepath)

def train(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    train_data, num_labels = load_data(args.train, 'train')
    dev_data = load_data(args.dev, 'valid')

    train_dataset = SentimentDataset(train_data, args)
    dev_dataset = SentimentDataset(dev_data, args)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=train_dataset.collate_fn)
    dev_loader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=dev_dataset.collate_fn)

    config = SimpleNamespace(
        hidden_dropout_prob=args.hidden_dropout_prob,
        num_labels=num_labels,
        hidden_size=768,
        data_dir='.',
        fine_tune_mode=args.fine_tune_mode
    )

    model = GPT2SentimentClassifier(config).to(device)

    # ✅ Discriminative Learning Rates 설정
    lr = args.lr
    param_groups = [
        {"params": [p for n, p in model.named_parameters() if "gpt" in n], "lr": lr * 0.5},
        {"params": [p for n, p in model.named_parameters() if "classifier" in n], "lr": lr}
    ]
    optimizer = AdamW(param_groups)

    # ✅ Slanted Triangular LR Scheduler
    total_steps = len(train_loader) * args.epochs
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    best_dev_acc = 0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            b_ids = batch['token_ids'].to(device)
            b_mask = batch['attention_mask'].to(device)
            b_labels = batch['labels'].to(device)

            optimizer.zero_grad()
            logits = model(b_ids, b_mask)
            loss = F.cross_entropy(logits, b_labels.view(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        train_acc, train_f1 = model_eval(train_loader, model, device)
        dev_acc, dev_f1 = model_eval(dev_loader, model, device)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        print(f"Epoch {epoch} | Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f} | Dev Acc: {dev_acc:.4f}")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine_tune_mode", type=str, choices=["last-linear-layer", "full-model"], default="last-linear-layer")
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train", type=str, default='data/ids-sst-train.csv')
    parser.add_argument("--dev", type=str, default='data/ids-sst-dev.csv')
    parser.add_argument("--test", type=str, default='data/ids-sst-test.csv')
    parser.add_argument("--filepath", type=str, default='sst-classifier.pt')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    train(args)
    test(args)
