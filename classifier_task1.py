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
        super().__init__()
        self.num_labels = config.num_labels
        self.gpt = GPT2Model.from_pretrained()
        assert config.fine_tune_mode in ["last-linear-layer", "full-model"]
        for param in self.gpt.parameters():
            param.requires_grad = (config.fine_tune_mode == "full-model")
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, self.num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.gpt(input_ids, attention_mask)
        last_hidden_state = outputs['last_hidden_state']
        last_non_pad_idx = attention_mask.sum(dim=1) - 1
        pooled_output = last_hidden_state[torch.arange(last_hidden_state.size(0)), last_non_pad_idx]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# ✅ Gradual Unfreezing
def gradual_unfreeze(model, current_epoch, total_epochs):
    layers = list(model.gpt.children())
    if not layers:
        return
    num_layers = len(layers)
    num_layers_to_unfreeze = int((current_epoch + 1) / total_epochs * num_layers)
    for i in range(num_layers_to_unfreeze):
        for param in layers[i].parameters():
            param.requires_grad = True

# ✅ Discriminative Learning Rates
def get_discriminative_lr_params(model, base_lr):
    return [
        {'params': model.gpt.parameters(), 'lr': base_lr / 10},
        {'params': model.classifier.parameters(), 'lr': base_lr}
    ]

# ✅ Slanted Triangular Learning Rate Scheduler
def get_stlr_scheduler(optimizer, total_steps, cut_frac=0.1):
    return get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(total_steps * cut_frac),
        num_training_steps=total_steps
    )

def train(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    train_data = load_data(args.train)
    dev_data = load_data(args.dev)

    train_dataset = SentimentDataset(train_data, args)
    dev_dataset = SentimentDataset(dev_data, args)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                              collate_fn=train_dataset.collate_fn)
    dev_loader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                            collate_fn=dev_dataset.collate_fn)

    config = SimpleNamespace(hidden_dropout_prob=args.hidden_dropout_prob, num_labels=5, hidden_size=768,
                             fine_tune_mode=args.fine_tune_mode)
    model = GPT2SentimentClassifier(config).to(device)

    total_steps = len(train_loader) * args.epochs
    optimizer = AdamW(get_discriminative_lr_params(model, args.lr))
    scheduler = get_stlr_scheduler(optimizer, total_steps)

    best_dev_acc = 0
    for epoch in range(args.epochs):
        gradual_unfreeze(model, epoch, args.epochs)  # ✅ Gradual Unfreezing
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f'Training Epoch {epoch}', disable=TQDM_DISABLE):
            b_ids = batch['token_ids'].to(device)
            b_mask = batch['attention_mask'].to(device)
            b_labels = batch['labels'].to(device)

            optimizer.zero_grad()
            logits = model(b_ids, b_mask)
            loss = F.cross_entropy(logits, b_labels)
            loss.backward()
            optimizer.step()
            scheduler.step()  # ✅ Scheduler step
            total_loss += loss.item()

        dev_acc, dev_f1 = model_eval(dev_loader, model, device)
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            torch.save(model.state_dict(), args.save_model_path)
        print(f"Epoch {epoch} | Loss: {total_loss/len(train_loader):.4f} | Dev Acc: {dev_acc:.4f} | Dev F1: {dev_f1:.4f}")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--fine_tune_mode", choices=["last-linear-layer", "full-model"], default="full-model")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    seed_everything()
    args.train = "data/ids-sst-train.csv"
    args.dev = "data/ids-sst-dev.csv"
    args.save_model_path = "gpt2_sentiment_sst.pt"
    train(args)
