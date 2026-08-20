import torch.nn as nn
from src.utils.config import GPTConfig
from src.models.embedding import GPTEmbedding
from src.models.block import DecoderBlock
from src.models.layer_norm import LayerNorm
from src.utils.causal_mask import create_causal_mask
import math


class GPT2(nn.Module):
    def __init__(self, config:GPTConfig):
        """Initialize the component and its configuration."""
        super().__init__()   
        self.residual_std = 0.02 / math.sqrt(2 * config.num_layers)

        self.embedding = GPTEmbedding(config)
        self.layers = nn.ModuleList(
            [
                DecoderBlock(config) for _ in range(config.num_layers)
                ]
        )
        self.norm = LayerNorm(config)

        self.output_projection  = nn.Linear(config.d_model, config.vocab_size, bias=config.bias)

        self.apply(self._init_weights)

        self._init_residual_weights()

        self.output_projection.weight = (self.embedding.token_embedding.weight)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, LayerNorm):
            nn.init.ones_(module.gamma)
            nn.init.zeros_(module.beta)

    def _init_residual_weights(self):
        for layer in self.layers:
            nn.init.normal_(layer.attn.out_proj.weight, mean=0.0, std=self.residual_std)
            if layer.attn.out_proj.bias is not None:
                nn.init.zeros_(layer.attn.out_proj.bias)

            nn.init.normal_(
                layer.ffn.fc2.weight,
                mean=0.0,
                std=self.residual_std
            )

            if layer.ffn.fc2.bias is not None:
                nn.init.zeros_(layer.ffn.fc2.bias)
            

    def forward(self, input_ids):

        """ 
        Args: 
            input_ids: (batch_size, seq_len) 
        Returns: 
            logits: (batch_size, seq_len, vocab_size) 
        """
        seq_len = input_ids.size(1)

        mask = create_causal_mask( seq_len, device=input_ids.device)

        x = self.embedding(input_ids)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)

        logits = self.output_projection(x)

        return logits