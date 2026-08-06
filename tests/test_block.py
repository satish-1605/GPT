
import torch

from src.utils.config import GPTConfig
from src.models.block import DecoderBlock

def test_block():
    config = GPTConfig()
    block = DecoderBlock(config)

    batch_size = 2
    seq_len = 10

    x = torch.randn(
        batch_size,
        seq_len, 
        config.d_model
    )
    output = block(x)

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
    test_block()