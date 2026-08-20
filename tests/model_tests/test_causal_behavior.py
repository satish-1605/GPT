import torch

from src.models.gpt2 import GPT2
from src.utils.config import GPTConfig


def test_causal_behavior():

    config = GPTConfig()

    model = GPT2(config)

    model.eval()

    # Two sequences that differ only in the future
    input_1 = torch.tensor([
        [10, 20, 30, 40]
    ])

    input_2 = torch.tensor([
        [10, 20, 99, 40]
    ])

    with torch.no_grad():

        logits_1 = model(input_1)

        logits_2 = model(input_2)

    # Position 0 cannot see positions 1, 2, 3
    assert torch.allclose(
        logits_1[:, 0, :],
        logits_2[:, 0, :],
        atol=1e-6
    )

    # Position 1 cannot see positions 2, 3
    assert torch.allclose(
        logits_1[:, 1, :],
        logits_2[:, 1, :],
        atol=1e-6
    )

    print("✅ Causal masking test passed!")


if __name__ == "__main__":
    test_causal_behavior()