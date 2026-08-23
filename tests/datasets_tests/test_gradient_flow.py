import torch

from src.models.gpt import GPT2
from src.utils.config import GPTConfig


def test_gradient_flow():

    config = GPTConfig()
    model = GPT2(config)

    model.train()

    batch_size = 2
    seq_len = 10

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (batch_size, seq_len),
        dtype=torch.long
    )

    # Forward
    logits = model(input_ids)

    # Simple scalar loss
    loss = logits.mean()

    # Backward
    loss.backward()

    # --------------------------------------------------
    # Check gradients
    # --------------------------------------------------

    assert model.embedding.token_embedding.weight.grad is not None

    assert model.layers[0].attn.wq.weight.grad is not None

    assert model.layers[0].attn.out_proj.weight.grad is not None

    assert model.layers[0].ffn.fc1.weight.grad is not None

    assert model.layers[0].ffn.fc2.weight.grad is not None

    assert model.norm.gamma.grad is not None

    assert model.output_projection.weight.grad is not None

    # --------------------------------------------------
    # Check NaN / Inf gradients
    # --------------------------------------------------

    for name, parameter in model.named_parameters():

        if parameter.grad is not None:

            assert not torch.isnan(
                parameter.grad
            ).any(), f"NaN gradient: {name}"

            assert not torch.isinf(
                parameter.grad
            ).any(), f"Inf gradient: {name}"

    print("=" * 60)
    print("Gradient flow       : ✅")
    print("NaN gradients       : ✅")
    print("Inf gradients       : ✅")
    print("Backward pass       : ✅")
    print("✅ Gradient flow test passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_gradient_flow()