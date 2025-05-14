import torch
from torch import nn
from transformers import GPT2Model as OpenAIGPT2Model

from config import GPT2Config
from models.base_gpt import GPTPreTrainedModel
from modules.gpt2_layer import GPT2Layer
from utils import get_extended_attention_mask


class GPT2Model(GPTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # Embedding layers.
        self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)

        position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
        self.register_buffer('position_ids', position_ids)

        # GPT-2 layers.
        self.gpt_layers = nn.ModuleList([GPT2Layer(config) for _ in range(config.num_hidden_layers)])

        # [CLS] 토큰 변환.
        self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_af = nn.Tanh()

        # Final layer norm.
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.init_weights()

    def embed(self, input_ids):
        input_shape = input_ids.size()
        seq_length = input_shape[1]

        inputs_embeds = self.word_embedding(input_ids)  # [bs, seq_len, hidden_size]

        pos_ids = self.position_ids[:, :seq_length]
        pos_embeds = self.pos_embedding(pos_ids)  # [1, seq_len, hidden_size]

        embeddings = inputs_embeds + pos_embeds
        embeddings = self.embed_dropout(embeddings)

        return embeddings

    def encode(self, hidden_states, attention_mask):
        extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, self.dtype)

        for i, layer_module in enumerate(self.gpt_layers):
            hidden_states = layer_module(hidden_states, extended_attention_mask)

        return hidden_states

    def forward(self, input_ids, attention_mask):
        embedding_output = self.embed(input_ids=input_ids)

        sequence_output = self.encode(embedding_output, attention_mask=attention_mask)
        sequence_output = self.final_layer_norm(sequence_output)

        last_non_pad_idx = attention_mask.sum(dim=1) - 1
        last_token = sequence_output[torch.arange(sequence_output.shape[0]), last_non_pad_idx]

        return {'last_hidden_state': sequence_output, 'last_token': last_token}

    def hidden_state_to_token(self, hidden_state):
        return torch.matmul(hidden_state, self.word_embedding.weight.t())

    @classmethod
    def from_pretrained(cls, model='gpt2', d=768, l=12, num_heads=12):
        gpt_model = OpenAIGPT2Model.from_pretrained(model).eval()
        our_model = GPT2Model(GPT2Config(hidden_size=d, num_hidden_layers=l, num_attention_heads=num_heads,
                                          intermediate_size=d * 3)).eval()

        our_model.word_embedding.load_state_dict(gpt_model.wte.state_dict())
        our_model.pos_embedding.load_state_dict(gpt_model.wpe.state_dict())

        for i in range(l):
            l = our_model.gpt_layers[i]
            l.self_attention.query.weight.data = gpt_model.state_dict()[f'h.{i}.attn.c_attn.weight'][:, :d].T
            l.self_attention.query.bias.data = gpt_model.state_dict()[f'h.{i}.attn.c_attn.bias'][:d]
            l.self_attention.key.weight.data = gpt_model.state_dict()[f'h.{i}.attn.c_attn.weight'][:, d:d * 2].T
            l.self_attention.key.bias.data = gpt_model.state_dict()[f'h.{i}.attn.c_attn.bias'][d:d * 2]
            l.self_attention.value.weight.data = gpt_model.state_dict()[f'h.{i}.attn.c_attn.weight'][:, d * 2:].T
            l.self_attention.value.bias.data = gpt_model.state_dict()[f'h.{i}.attn.c_attn.bias'][d * 2:]

            l.attention_dense.weight.data = gpt_model.state_dict()[f'h.{i}.attn.c_proj.weight'].T
            l.attention_dense.bias.data = gpt_model.state_dict()[f'h.{i}.attn.c_proj.bias']

            l.attention_layer_norm.weight.data = gpt_model.state_dict()[f'h.{i}.ln_1.weight']
            l.attention_layer_norm.bias.data = gpt_model.state_dict()[f'h.{i}.ln_1.bias']

            l.interm_dense.weight.data = gpt_model.state_dict()[f'h.{i}.mlp.c_fc.weight'].T
            l.interm_dense.bias.data = gpt_model.state_dict()[f'h.{i}.mlp.c_fc.bias']
            l.out_dense.weight.data = gpt_model.state_dict()[f'h.{i}.mlp.c_proj.weight'].T
            l.out_dense.bias.data = gpt_model.state_dict()[f'h.{i}.mlp.c_proj.bias']

            l.out_layer_norm.weight.data = gpt_model.state_dict()[f'h.{i}.ln_2.weight']
            l.out_layer_norm.bias.data = gpt_model.state_dict()[f'h.{i}.ln_2.bias']

        our_model.final_layer_norm.weight.data = gpt_model.state_dict()['ln_f.weight']
        our_model.final_layer_norm.bias.data = gpt_model.state_dict()['ln_f.bias']

        return our_model
