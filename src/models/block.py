import torch.nn as nn
from src.utils.config import GPTConfig
from src.models.attention import MultiHeadAttention
from src.models.layer_norm import LayerNorm
from src.models.mlp import FeedForward



class DecoderBlock(nn.Module):
    def __init__(self, config:GPTConfig):
        """Initialize the component and its configuration."""
        super().__init__()
        self.norm1 = LayerNorm(config)
        self.attn = MultiHeadAttention(config)
        self.norm2 = LayerNorm(config)       
        self.ffn = FeedForward(config)

    def forward(self, x, mask):

        x = x + self.attn(self.norm1(x), mask)

        x = x + self.ffn(self.norm2(x))

        return x





