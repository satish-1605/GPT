import torch
import torch.nn as nn
from src.utils.config import GPTConfig

class LayerNorm(nn.Module):
    """
        Custom implementation of Layer Normalization.
        Equivalent to nn.LayerNorm(normalized_shape=d_model).
    """
    def __init__(self, config:GPTConfig):
        """Initialize the component and its configuration."""
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(config.d_model))
        self.beta = nn.Parameter(torch.zeros(config.d_model))
        self.eps = config.eps

    def forward(self, x):     
        """Run a forward pass through this component."""
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        normalized = (x - mean) * torch.rsqrt(var + self.eps)
        norm_output = self.gamma * normalized + self.beta
        return norm_output

