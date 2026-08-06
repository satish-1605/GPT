from src.utils.config import GPTConfig
from src.models.mlp import FeedForward
import torch


def test_ffn():
    config = GPTConfig()
    ffn = FeedForward(config)

    batch_size = 2
    seq_len = 10

    x = torch.randn(
            batch_size,
            seq_len, 
            config.d_model
        )
    output = ffn(x)

    print("Input :", x.shape)
    print("Output:", output.shape)

    assert output.shape == x.shape
    assert not torch.isnan(output).any()

    print("✅ FeedForward test passed!")

if __name__ == "__main__":
    test_ffn()