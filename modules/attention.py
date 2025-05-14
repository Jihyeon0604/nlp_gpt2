import torch

from einops import rearrange
from torch import nn


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # key, value, query에 대한 선형변환 layer 초기화.
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 트랜스포머 원래 구현에 따라 normalized attention scores에 적용되는 dropout.
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transform(self, x, linear_layer):
        # hidden_state (x) 를 사영하기 위해 k, v, q의 해당 linear_layer가 사용된다.
        proj = linear_layer(x)
        # 여러 헤드 생성: [batch, seq_len, hidden_size] -> [batch, num_heads, seq_len, head_dim]
        proj = rearrange(proj, 'b t (h d) -> b h t d', h=self.num_attention_heads)
        return proj

    def attention(self, key, query, value, attention_mask):
        """
        key, query, value: [bs, num_heads, seq_len, head_dim]
        attention_mask: [bs, 1, 1, seq_len]
        """
        # 1. Scaled Dot-Product Attention Score
        dk = query.shape[-1]
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(dk, dtype=torch.float32, device=query.device))
        # scores: [bs, num_heads, seq_len, seq_len]

        # 2. Apply Causal Mask (prevent attending to future tokens)
        seq_len = scores.size(-1)
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=scores.device)).unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))

        # 3. Apply Attention Mask (if provided)
        if attention_mask is not None:
            scores = scores + attention_mask  # 이미 올바른 형태로 주어짐

        # 4. Softmax + Dropout
        attn_probs = torch.softmax(scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # 5. Compute Weighted Sum of Values
        context = torch.matmul(attn_probs, value)  # [bs, num_heads, seq_len, head_dim]

        # 6. Concatenate Multiple Heads
        context = rearrange(context, 'b h t d -> b t (h d)')  # [bs, seq_len, hidden_size]

        return context

    def forward(self, hidden_states, attention_mask):
        """
        hidden_states: [bs, seq_len, hidden_size]
        attention_mask: [bs, 1, 1, seq_len]
        output: [bs, seq_len, hidden_size]
        """
        key_layer = self.transform(hidden_states, self.key)
        value_layer = self.transform(hidden_states, self.value)
        query_layer = self.transform(hidden_states, self.query)

        attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
        return attn_value
