import torch
from torch import nn
from config import GPT2Config
from models.base_gpt import GPTPreTrainedModel
from modules.lora_gpt2_layer import LoRAGPT2Layer
from utils import get_extended_attention_mask

class GPT2Model(GPTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)

        position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
        self.register_buffer('position_ids', position_ids)

        # ✅ LoRA 적용된 레이어 사용
        self.gpt_layers = nn.ModuleList([LoRAGPT2Layer(config) for _ in range(config.num_hidden_layers)])

        self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_af = nn.Tanh()

        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.init_weights()

    def embed(self, input_ids):
        input_shape = input_ids.size()
        seq_length = input_shape[1]
        inputs_embeds = self.word_embedding(input_ids)
        pos_ids = self.position_ids[:, :seq_length]
        pos_embeds = self.pos_embedding(pos_ids)
        embeddings = inputs_embeds + pos_embeds
        embeddings = self.embed_dropout(embeddings)
        return embeddings

    def encode(self, hidden_states, attention_mask):
        extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, self.dtype)
        for layer_module in self.gpt_layers:
            hidden_states = layer_module(hidden_states, extended_attention_mask)
        return hidden_states

    def forward(self, input_ids, attention_mask):
        embedding_output = self.embed(input_ids=input_ids)
        sequence_output = self.encode(embedding_output, attention_mask=attention_mask)
        sequence_output = self.final_layer_norm(sequence_output)

        last_non_pad_idx = attention_mask.sum(dim=1) - 1
        last_token = sequence_output[torch.arange(sequence_output.shape[0]), last_non_pad_idx]

        return {'last_hidden_state': sequence_output, 'last_token': last_token}
