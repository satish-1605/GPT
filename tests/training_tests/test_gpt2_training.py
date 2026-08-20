import pytest
import torch

from src.models.gpt2 import GPT2
from src.utils.config import GPTConfig

from src.training.optimizer import create_optimizer
from src.training.gradient import clip_gradients
from src.training.trainer import train_one_epoch

from src.losses.loss import calculate_loss
from src.datasets.data_pipeline import get_train_val_test_loaders
config = GPTConfig()

config.vocab_size = 500
config.context_length = 32
config.num_heads = 2
config.num_layers = 2
config.d_model = 32
config.ff_hidden_dim = 128
config.batch_size = 2

@pytest.fixture
def training_components():

    

    device = torch.device(config.device)

    train_loader, _, _ = get_train_val_test_loaders(config)

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

def test_single_training_step(training_components):

    (
        model,
        optimizer,
        train_loader,
        device,
        config,
    ) = training_components
    model.train()

    input_ids, target_ids = next(iter(train_loader))

    input_ids = input_ids.to(device)
    target_ids = target_ids.to(device)

    optimizer.zero_grad(set_to_none=True)

    logits = model(input_ids)

    loss = calculate_loss(
        logits,
        target_ids,
    )

    assert torch.isfinite(loss)

    loss.backward()

    grad_norm = clip_gradients(
        model,
        config.max_grad_norm,
    )

    assert torch.isfinite(
        torch.tensor(grad_norm)
    )

    optimizer.step()

def test_parameters_update(training_components):

    (
        model,
        optimizer,
        train_loader,
        device,
        config,
    ) = training_components
    model.train()

    input_ids, target_ids = next(iter(train_loader))

    input_ids = input_ids.to(device)
    target_ids = target_ids.to(device)

    # Pick one parameter
    parameter = next(model.parameters())

    before = parameter.detach().clone()

    optimizer.zero_grad(set_to_none=True)

    logits = model(input_ids)

    loss = calculate_loss(
        logits,
        target_ids,
    )

    loss.backward()

    clip_gradients(
        model,
        config.max_grad_norm,
    )

    optimizer.step()

    after = parameter.detach()

    assert not torch.equal(
        before,
        after,
    )

def test_train_one_epoch(training_components):

    (
        model,
        optimizer,
        train_loader,
        device,
        config,
    ) = training_components

    avg_loss, global_step = train_one_epoch(
        loader=train_loader,
        model=model,
        optimizer=optimizer,
        device=device,
        config=config,
        max_steps=1,
    )

    assert torch.isfinite(
        torch.tensor(avg_loss)
    )

    assert global_step == 1

def test_max_steps(training_components):

    (
        model,
        optimizer,
        train_loader,
        device,
        config,
    ) = training_components
    max_steps = 3

    avg_loss, global_step = train_one_epoch(
        loader=train_loader,
        model=model,
        optimizer=optimizer,
        device=device,
        config=config,
        global_step=0,
        max_steps=max_steps,
    )

    assert torch.isfinite(
        torch.tensor(avg_loss)
    )

    assert global_step == max_steps

def test_global_step_continuation(training_components):

    (
        model,
        optimizer,
        train_loader,
        device,
        config,
    ) = training_components

    initial_step = 10
    max_steps = 13

    avg_loss, global_step = train_one_epoch(
        loader=train_loader,
        model=model,
        optimizer=optimizer,
        device=device,
        config=config,
        global_step=initial_step,
        max_steps=max_steps,
    )

    assert global_step == max_steps

def create_training_components():

    config = GPTConfig()

    device = torch.device(
        config.device
    )

    tokenizer = BPETokenizer.from_pretrained(
        config.load_dir
    )

    # Use your existing Phase C loader
    train_loader, _, _ = get_train_val_test_loaders(
        config
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
