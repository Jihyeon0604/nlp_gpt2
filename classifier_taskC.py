#!/usr/bin/env python3

import random, numpy as np, argparse, csv
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
from optimizer import AdamW
from models.gpt2_lora import GPT2Model  # ✅ LoRA 적용된 GPT2
from model_utils import SentimentDataset, SentimentTestDataset, load_data, save_model, model_eval, model_test_eval

TQDM_DISABLE = False

def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class LoRALinear(torch.nn.Module):
    def __init__(self, in_features, out_features, r=8, lora_alpha=16, dropout=0.1):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.dropout = torch.nn.Dropout(dropout)
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = torch.nn.Parameter(torch.zeros(out_features))
        self.lora_A = torch.nn.Linear(in_features, r, bias=False)
        self.lora_B = torch.nn.Linear(r, out_features, bias=False)
        self.scaling = lora_alpha / r

    def forward(self, x):
        result = F.linear(x, self.weight, self.bias)
        if self.training:
            x = self.dropout(x)
        lora_result = self.lora_B(self.lora_A(x))
        return result + self.scaling * lora_result


class GPT2SentimentClassifier(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels
        self.gpt = GPT2Model(config)
        hidden_size = config.hidden_size
        self.score = LoRALinear(hidden_size, self.num_labels, r=8, lora_alpha=16, dropout=0.1)

    def forward(self, input_ids, attention_mask):
        outputs = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs['last_hidden_state']
        last_token_indices = attention_mask.sum(dim=1) - 1
        last_token_output = last_hidden[torch.arange(last_hidden.size(0)), last_token_indices]
        logits = self.score(last_token_output)
        return logits


def train(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    train_data, num_labels = load_data(args.train, 'train')
    dev_data = load_data(args.dev, 'valid')
    train_dataset = SentimentDataset(train_data, args)
    dev_dataset = SentimentDataset(dev_data, args)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=dev_dataset.collate_fn)

    args.num_labels = num_labels
    args.hidden_size = 768
    model = GPT2SentimentClassifier(args).to(device)

    # ✅ 전체 fine-tuning
    for param in model.parameters():
        param.requires_grad = True

    optimizer = AdamW(model.parameters(), lr=args.lr)
    best_dev_acc = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            b_ids, b_mask, b_labels = batch['token_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
            optimizer.zero_grad()
            logits = model(b_ids, b_mask)
            loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_acc, train_f1, *_ = model_eval(train_dataloader, model, device)
        dev_acc, dev_f1, *_ = model_eval(dev_dataloader, model, device)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, args, args.filepath)

        print(f"Epoch {epoch}: train loss :: {train_loss:.3f}, train acc :: {train_acc:.3f}, train F1 :: {train_f1:.3f}, dev acc :: {dev_acc:.3f}, dev F1 :: {dev_f1:.3f}")


def test(args):
    from types import SimpleNamespace
    import torch.serialization
    import numpy.core.multiarray as multiarray
    torch.serialization.add_safe_globals([SimpleNamespace, multiarray._reconstruct, np.ndarray])
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath, weights_only=False)
        config = saved['model_config']
        model = GPT2SentimentClassifier(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)

        dev_data = load_data(args.dev, 'valid')
        test_data = load_data(args.test, 'test')
        dev_dataloader = DataLoader(SentimentDataset(dev_data, args), shuffle=False, batch_size=args.batch_size, collate_fn=SentimentDataset(dev_data, args).collate_fn)
        test_dataloader = DataLoader(SentimentTestDataset(test_data, args), shuffle=False, batch_size=args.batch_size, collate_fn=SentimentTestDataset(test_data, args).collate_fn)

        dev_acc, dev_f1, dev_pred, *_ = model_eval(dev_dataloader, model, device)
        print('DONE DEV')

        test_pred, test_sents, test_sent_ids = model_test_eval(test_dataloader, model, device)
        print('DONE Test')

        with open(args.dev_out, "w+") as f:
            print(f"dev acc :: {dev_acc :.3f}")
            f.write("id \t Predicted_Sentiment \n")
            for p, s in zip([x[2] for x in dev_data], dev_pred):
                f.write(f"{p}, {s} \n")

        with open(args.test_out, "w+") as f:
            f.write("id \t Predicted_Sentiment \n")
            for p, s in zip([x[1] for x in test_data], test_pred):
                f.write(f"{p}, {s} \n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine-tune-mode", type=str, choices=('last-linear-layer', 'full-model'), default="full-model")
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # SST
    sst_batch_size = args.batch_size if args.batch_size > 0 else 64
    config = SimpleNamespace(
        filepath='sst-classifier.pt',
        lr=args.lr,
        use_gpu=args.use_gpu,
        epochs=args.epochs,
        batch_size=sst_batch_size,
        hidden_dropout_prob=args.hidden_dropout_prob,
        train='data/ids-sst-train.csv',
        dev='data/ids-sst-dev.csv',
        test='data/ids-sst-test-student.csv',
        fine_tune_mode=args.fine_tune_mode,
        dev_out='predictions/' + args.fine_tune_mode + '-sst-dev-out.csv',
        test_out='predictions/' + args.fine_tune_mode + '-sst-test-out.csv',
        name_or_path='gpt2',
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        max_position_embeddings=tokenizer.model_max_length,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=2304,
        layer_norm_eps=1e-5,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
    )

    train(config)
    print('Evaluating on SST...')
    test(config)

    # CFIMDB
    cfimdb_batch_size = args.batch_size if args.batch_size > 0 else 8
    config.filepath = 'cfimdb-classifier.pt'
    config.train = 'data/ids-cfimdb-train.csv'
    config.dev = 'data/ids-cfimdb-dev.csv'
    config.test = 'data/ids-cfimdb-test-student.csv'
    config.batch_size = cfimdb_batch_size
    config.dev_out = 'predictions/' + args.fine_tune_mode + '-cfimdb-dev-out.csv'
    config.test_out = 'predictions/' + args.fine_tune_mode + '-cfimdb-test-out.csv'

    train(config)
    print('Evaluating on cfimdb...')
    test(config)
