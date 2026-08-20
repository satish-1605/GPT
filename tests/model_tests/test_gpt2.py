from src.models.gpt2 import GPT2
from src.utils.config import GPTConfig
import torch


def test_gpt2_forward():

    config = GPTConfig()
    model = GPT2(config)

    batch_size = 2
    seq_len = 10

    # Token IDs
    input_ids = torch.randint(
        0,
        config.vocab_size,
        (batch_size, seq_len),
        dtype=torch.long
    )

    # Forward pass
    logits = model(input_ids)

    print("=" * 60)
    print("Input Shape :", input_ids.shape)
    print("Output Shape:", logits.shape)
    print("Output dtype:", logits.dtype)
    print("=" * 60)

    # --------------------------------------------------
    # Shape test
    # --------------------------------------------------

    assert input_ids.shape == (
        batch_size,
        seq_len
    )

    assert logits.shape == (
        batch_size,
        seq_len,
        config.vocab_size
    )

    # --------------------------------------------------
    # Dtype test
    # --------------------------------------------------

    assert logits.dtype == torch.float32

    # --------------------------------------------------
    # Numerical stability
    # --------------------------------------------------

    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()

    # --------------------------------------------------
    # Weight tying
    # --------------------------------------------------

    assert (
        model.output_projection.weight
        is model.embedding.token_embedding.weight
    )

    # --------------------------------------------------
    # Backward test
    # --------------------------------------------------

    loss = logits.mean()
    loss.backward()

    print("Weight tying : ✅")
    print("NaN check    : ✅")
    print("Inf check    : ✅")
    print("Backward     : ✅")
    print("✅ GPT-2 forward test passed!")


if __name__ == "__main__":
    test_gpt2_forward()