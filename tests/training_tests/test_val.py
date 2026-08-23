import torch
from src.training.validate import validate
import pytest
from src.utils.config import GPTConfig
from src.datasets.dataset import GPTDataset
from src.datasets.dataloader import create_dataloader
from src.models.gpt import GPT2
from src.training.optimizer import create_optimizer


@pytest.fixture
def training_components():

    config = GPTConfig()

    device = torch.device(config.device)

    token_stream = list(range(100))

    dataset = GPTDataset(
        token_stream=token_stream,
        context_length=32,
        stride=32,
    )

    train_loader = create_dataloader(
        dataset=dataset,
        batch_size=2,
        shuffle=False,
    )

    model = GPT2(config).to(device)

    optimizer = create_optimizer(
        model=model,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2),
        eps=config.adam_eps,
    )

    return (
        model,
        optimizer,
        train_loader,
        device,
        config,
    )

def test_validate_returns_finite_loss(training_components):

    model, optimizer, train_loader, device, config = training_components

    val_loss = validate(
        loader=train_loader,
        model=model,
        device=device,
    )

    assert torch.isfinite(
        torch.tensor(val_loss)
    )

def test_validate_does_not_update_parameters(training_components):

    model, optimizer, train_loader, device, config = training_components

    parameter = next(model.parameters())

    before = parameter.detach().clone()

    validate(
        loader=train_loader,
        model=model,
        device=device,
    )

    after = parameter.detach()

    assert torch.equal(
        before,
        after,
    )

def test_validate_does_not_create_gradients(training_components):

    model, optimizer, train_loader, device, config = training_components

    model.zero_grad(set_to_none=True)

    validate(
        loader=train_loader,
        model=model,
        device=device,
    )

    for parameter in model.parameters():
        assert parameter.grad is None

def test_validate_sets_eval_mode(training_components):

    model, optimizer, train_loader, device, config = training_components

    model.train()

    assert model.training

    validate(
        loader=train_loader,
        model=model,
        device=device,
    )

    assert not model.training