import torch
import torch.nn as nn
from models.gpt2 import GPT2Model
from modules.lora import LoRALinear

class GPT2WithLoRAHybrid(nn.Module):
    def __init__(self, config, lora_r=8, lora_alpha=32):
        super().__init__()
        self.gpt = GPT2Model.from_pretrained()
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)

        self.lora_layers = nn.ModuleList()
        for layer in self.gpt.gpt_layers:
            lora = nn.ModuleDict({
                "query": LoRALinear(self.hidden_size, self.hidden_size, r=lora_r, alpha=lora_alpha),
                "value": LoRALinear(self.hidden_size, self.hidden_size, r=lora_r, alpha=lora_alpha)
            })
            self.lora_layers.append(lora)

        self._freeze_except_top_layers()

    def _freeze_except_top_layers(self):
        for param in self.gpt.parameters():
            param.requires_grad = False

        # top 2 transformer layers (10, 11)만 학습 허용
        for i in range(len(self.gpt.gpt_layers)):
            if i >= 10:
                for param in self.gpt.gpt_layers[i].parameters():
                    param.requires_grad = True

        for lora_layer in self.lora_layers:
            for param in lora_layer.parameters():
                param.requires_grad = True

        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        gpt_outputs = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
        hidden = gpt_outputs["last_hidden_state"]

        for i, layer in enumerate(self.lora_layers):
            hidden = hidden + layer["query"](hidden) + layer["value"](hidden)

        last_token = gpt_outputs["last_token"]
        pooled = self.dropout(last_token)
        logits = self.classifier(pooled)
        return logits
