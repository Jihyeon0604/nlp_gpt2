import torch
from einops import rearrange
from torch import nn
from modules.lora import LoRALinear  # 기존 LoRA 모듈 재사용

class LoRACausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # LoRA 적용 (query, value만)
        self.query = LoRALinear(config.hidden_size, self.all_head_size, r=8, alpha=32)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = LoRALinear(config.hidden_size, self.all_head_size, r=8, alpha=32)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transform(self, x, linear_layer):
        proj = linear_layer(x)
        proj = rearrange(proj, 'b t (h d) -> b h t d', h=self.num_attention_heads)
        return proj

    def attention(self, key, query, value, attention_mask):
        dk = query.shape[-1]
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(dk, dtype=torch.float32, device=query.device))
        seq_len = scores.size(-1)
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=scores.device)).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))
        if attention_mask is not None:
            scores = scores + attention_mask
        attn_probs = torch.softmax(scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        context = torch.matmul(attn_probs, value)
        context = rearrange(context, 'b h t d -> b t (h d)')
        return context

    def forward(self, hidden_states, attention_mask):
        key_layer = self.transform(hidden_states, self.key)
        value_layer = self.transform(hidden_states, self.value)
        query_layer = self.transform(hidden_states, self.query)
        attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
        return attn_value
