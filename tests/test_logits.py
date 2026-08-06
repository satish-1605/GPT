from src.models.gpt import GPT
import torch
from src.utils.config import GPTConfig
import torch.nn.functional as F

def test_gpt():
    config = GPTConfig()
    model = GPT(config)

    batch_size = 2
    seq_len = 10
    
    input_ids = torch.randint(
        0,
        config.vocab_size,
        (batch_size, seq_len)
    )
    output = model(input_ids)

    print("=" * 60)
    print("Logit Statistics")

    print("Min :", output.min().item())
    print("Max :", output.max().item())
    print("Mean:", output.mean().item())

    print("NaN :", torch.isnan(output).any().item())
    print("Inf :", torch.isinf(output).any().item())

    print("=" * 60)
    print("Input Shape :", input_ids.shape)
    print("Output Shape:", output.shape)
    print("Output dtype:", output.dtype)
    print("=" * 60)

    assert output.shape == (
        batch_size,
        seq_len,
        config.vocab_size
    )
    assert not torch.isnan(output).any()


    print("✅ GPT test passed!")

    print("=" * 60)

    print("Model Validation")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    print(total_params)
    print(trainable_params)

    print("=" * 60)
    
    print("One loss computaion")
    targets = torch.randint(
    0,
    config.vocab_size,
    (batch_size, seq_len))

    loss = F.cross_entropy(
    output.view(-1, config.vocab_size),
    targets.view(-1)
    )

    print(loss.item())

    print("=" * 60)
    
    print("Gradient check")
    loss.backward()
    print(model.embedding.token_embedding.weight.grad is not None)




if __name__ == "__main__":
    test_gpt()
