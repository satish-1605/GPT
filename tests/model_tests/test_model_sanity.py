import torch 
from src.models.gpt import GPT2
from src.utils.config import GPTConfig
def test_gpt2_sanity():

    config = GPTConfig()
    model = GPT2(config)

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (2, 10),
        dtype=torch.long
    )

    logits = model(input_ids)

    # Shape
    assert logits.shape == (
        2,
        10,
        config.vocab_size
    )

    # Numerical stability
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()

    # Weight tying
    assert (
        model.output_projection.weight
        is model.embedding.token_embedding.weight
    )

    # Backward
    loss = logits.mean()
    loss.backward()

    print("✅ GPT-2 sanity test passed!")


if __name__ == "__main__":
    test_gpt2_sanity()