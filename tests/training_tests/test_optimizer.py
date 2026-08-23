
import torch

from src.models.gpt import GPT2
from src.utils.config import GPTConfig
from src.training.optimizer import create_optimizer


def test_create_optimizer():

    config = GPTConfig()

    model = GPT2(config)

    optimizer = create_optimizer(
        model=model,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2),
        eps=config.adam_eps,
    )

    assert isinstance(optimizer, torch.optim.AdamW)

    assert optimizer.defaults["lr"] == config.learning_rate
    assert optimizer.defaults["weight_decay"] == config.weight_decay
    assert optimizer.defaults["betas"] == (
        config.beta1,
        config.beta2,
    )
    assert optimizer.defaults["eps"] == config.adam_eps

if __name__ == "__main__":
    test_create_optimizer()