import torch

from src.utils.config import GPTConfig
from src.models.embedding import GPTEmbedding


def test_embedding():

    # Configuration
    config = GPTConfig()

    # Embedding Layer
    embedding = GPTEmbedding(config)

    # Dummy input
    batch_size = 2
    seq_len = 10

    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, seq_len)
    )

    # Forward pass
    output = embedding(input_ids)

    print("=" * 50)
    print("Input Shape :", input_ids.shape)
    print("Output Shape:", output.shape)
    print("Output dtype:", output.dtype)
    print("=" * 50)


if __name__ == "__main__":
    test_embedding()