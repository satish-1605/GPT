import torch.nn as nn
from src.utils.config import GPTConfig



class FeedForward(nn.Module):
    def __init__(self, config:GPTConfig):
        """Initialize the component and its configuration."""
        super().__init__()

        self.fc1 = nn.Linear(config.model.d_model, config.model.ff_hidden_dim, bias=config.model.bias)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(config.model.ff_hidden_dim, config.model.d_model, bias=config.model.bias)

        self.dropout = nn.Dropout(config.model.dropout)

    def forward(self, x):
        """Run a forward pass through this component."""
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
