import torch
import math
import torch.nn as nn
from src.utils.config import GPTConfig

class MultiHeadAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        """Initialize the component and its configuration."""
        super().__init__()
        self.num_heads = config.num_heads

        assert config.d_model % self.num_heads == 0, (
            "d_model must be divisible by num_heads"
        )

        self.head_dim = config.d_model // config.num_heads

        self.wq = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.wk = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.wv = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        self.dropout = nn.Dropout(config.dropout)

    def split_heads(self, x):
        """
        (Batch size, Seq Len, Dim) -> (Batch size, num_Head, Seq, head_dim)
        """

        B, S, _ = x.size()

        x = x.view(B, S, self.num_heads, self.head_dim)

        return x.transpose(1,2)

    def scaled_dot_product_attention(self, q, k, v, mask):
        """
        q, k, v: (B, H, S, head_dim)
        """
        scores = q @ k.transpose(-2,-1)
        scores = scores / math.sqrt(self.head_dim)

        scores = scores.masked_fill(~mask, float("-inf"))

        attn = scores.softmax(dim=-1)
        attn = self.dropout(attn)

        output = attn @ v

        return output, attn        


    def merge_heads(self, x):
        """
        (B, H, S, head_dim) -> (B, S, D)
        """

        B, H, S, D = x.size()
        x = x.transpose(1,2).contiguous()

        return x.view(B, S, H * D)


    def forward(self, x, mask):
        """
            x: (B, S, d_model)
            mask: (1, 1, S, S) or (B, 1, S, S)
        """
        q = self.split_heads(self.wq(x))
        k = self.split_heads(self.wk(x))
        v = self.split_heads(self.wv(x))

        context, attn = self.scaled_dot_product_attention(q, k, v, mask)
        context = self.merge_heads(context)

        output = self.out_proj(context)

        return output