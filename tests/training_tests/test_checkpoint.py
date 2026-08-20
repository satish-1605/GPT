import torch
from src.training.validate import validate
import pytest
from src.utils.config import GPTConfig
from src.datasets.dataset import GPTDataset
from src.datasets.dataloader import create_dataloader
from src.models.gpt2 import GPT2
from src.training.optimizer import create_optimizer
from src.training.checkpoint import save_checkpoint, load_checkpoint

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


def test_save_checkpoint(tmp_path, training_components):

    model, optimizer, loader, device, config = training_components

    checkpoint_path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=2,
        global_step=100,
        best_val_loss=4.5,
        config=config,
    )

    assert checkpoint_path.exists()

def test_checkpoint_contents(tmp_path, training_components):

    model, optimizer, loader, device, config = training_components

    checkpoint_path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=2,
        global_step=100,
        best_val_loss=4.5,
        config=config,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint
    assert "epoch" in checkpoint
    assert "global_step" in checkpoint
    assert "best_val_loss" in checkpoint
    assert "config" in checkpoint

def test_load_checkpoint(
    tmp_path,
    training_components,
):
    model, optimizer, loader, device, config = training_components

    checkpoint_path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=3,
        global_step=150,
        best_val_loss=4.2,
        config=config,
    )

    new_model = GPT2(config).to(device)

    new_optimizer = create_optimizer(
        model=new_model,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2),
        eps=config.adam_eps,
    )

    epoch, global_step, best_val_loss = load_checkpoint(
        path=checkpoint_path,
        model=new_model,
        optimizer=new_optimizer,
        device=device,
    )

    assert epoch == 3
    assert global_step == 150
    assert best_val_loss == 4.2

def test_checkpoint_restores_weights(
    tmp_path,
    training_components,
):
    model, optimizer, loader, device, config = training_components

    checkpoint_path = tmp_path / "checkpoint.pt"

    original_weights = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=1,
        global_step=50,
        best_val_loss=5.0,
        config=config,
    )

    new_model = GPT2(config).to(device)

    new_optimizer = create_optimizer(
        model=new_model,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2),
        eps=config.adam_eps,
    )

    load_checkpoint(
        path=checkpoint_path,
        model=new_model,
        optimizer=new_optimizer,
        device=device,
    )

    for name, parameter in new_model.named_parameters():

        assert torch.equal(
            original_weights[name],
            parameter,
        )