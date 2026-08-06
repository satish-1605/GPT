import torch.nn as nn
import torch
from src.utils.config import GPTConfig

class GPTEmbedding(nn.Module):
    """
        GPT-1 Embedding Layer

        Combines:
            1. Token Embedding
            2. Learnable Positional Embedding
            3. Dropout

        Input:
            input_ids : (batch_size, seq_len)

        Output:
            embeddings : (batch_size, seq_len, d_model)
    """
    def __init__(self, config:GPTConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout= nn.Dropout(config.dropout)

    def forward(self, input_ids:torch.Tensor)-> torch.Tensor:
        """
        input_ids: Tensor of shape (batch_size, seq_len)
        Returns:
            embeddings: Tensor of shape (batch_size, seq_len, d_model)
        """
        seq_len = input_ids.size(1)

        positions = torch.arange(
            seq_len, 
            device=input_ids.device
                ).unsqueeze(0)

        token_embeddings = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(positions)

        embeddings = token_embeddings + position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings
