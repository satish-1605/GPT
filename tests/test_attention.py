
import torch

from src.utils.config import GPTConfig
from src.utils.causal_mask import create_causal_mask

from src.models.attention import MultiHeadAttention

def test_mha():
    config = GPTConfig()
    mha = MultiHeadAttention(config)

    batch_size = 2
    seq_len = 10

    x = torch.randn(
        batch_size,
        seq_len, 
        config.d_model
    )
    output = mha(x)

    print("=" * 60)
    print("Input Shape :", x.shape)
    print("Output Shape:", output.shape)
    print("Output dtype:", output.dtype)
    print("=" * 60)

    assert output.shape == (
        batch_size,
        seq_len,
        config.d_model
    )

    assert output.dtype == torch.float32

    assert not torch.isnan(output).any()

    print("✅ All tests passed!")

if __name__ == "__main__":
    test_mha()