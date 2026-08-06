import torch.nn as nn
from src.utils.config import GPTConfig
from src.models.embedding import GPTEmbedding
from src.models.block import DecoderBlock
from src.models.layer_norm import LayerNorm
from src.utils.causal_mask import create_causal_mask


class GPT(nn.Module):
    def __init__(self, config:GPTConfig):
        """Initialize the component and its configuration."""
        super().__init__()   

        self.embedding = GPTEmbedding(config)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(config) for _ in range(config.num_layers)
                ]
        )

        self.norm = LayerNorm(config)

        self.output_projection  = nn.Linear(config.d_model, config.vocab_size, bias=config.bias)

        # self.output_projection.weight = self.embedding.token_embedding.weight
        #can use when weight init is around means 0, std 0.02


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



