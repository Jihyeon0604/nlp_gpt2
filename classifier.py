import random, numpy as np, argparse, csv
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from sklearn.metrics import f1_score, accuracy_score
from models.gpt2 import GPT2Model
from optimizer import AdamW
from tqdm import tqdm
from types import SimpleNamespace

TQDM_DISABLE = False

def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
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
        labels = [x[1] for x in data]
        prompts = [f"Review: {sent}" for sent in sents]
        encoding = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])
        labels = torch.LongTensor(labels)
        return token_ids, attention_mask, labels

    def collate_fn(self, all_data):
        token_ids, attention_mask, labels = self.pad_data(all_data)
        return {'token_ids': token_ids, 'attention_mask': attention_mask, 'labels': labels}

def load_data(filename):
    data = []
    with open(filename, 'r') as fp:
        for record in csv.DictReader(fp, delimiter='\t'):
            sent = record['sentence'].lower().strip()
            label = int(record['sentiment'].strip())
            data.append((sent, label))
    return data

def model_eval(dataloader, model, device):
    model.eval()
    y_true, y_pred = [], []
    for batch in tqdm(dataloader, desc='eval', disable=TQDM_DISABLE):
        b_ids = batch['token_ids'].to(device)
        b_mask = batch['attention_mask'].to(device)
        b_labels = batch['labels'].to(device)

        logits = model(b_ids, b_mask)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        y_true.extend(b_labels.cpu().numpy())
        y_pred.extend(preds)

    f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)
    return acc, f1

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

    optimizer = AdamW(model.parameters(), lr=args.lr)
    best_dev_acc = 0

    for epoch in range(args.epochs):
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

    # ✅ SST-5 학습만 실행
    args.train = "data/ids-sst-train.csv"
    args.dev = "data/ids-sst-dev.csv"
    args.save_model_path = "gpt2_sentiment_sst.pt"
    train(args)
