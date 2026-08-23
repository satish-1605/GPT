import torch

from src.losses.loss import calculate_loss
from src.training.gradient import clip_gradients
from src.training.scheduler import get_learning_rate


def train_one_epoch(
    loader,
    model,
    optimizer,
    device,
    config,
    global_step: int = 0,
    max_steps: int | None = None,
):
    model.train()

    total_loss = 0.0
    num_batches = 0

    for input_ids, target_ids in loader:

        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)


        optimizer.zero_grad(set_to_none=True)

        logits = model(input_ids)

        loss = calculate_loss(
            logits,
            target_ids,
        )

        loss.backward()

        grad_norm = clip_gradients(
            model,
            config.training.max_grad_norm,
        )

        if max_steps is not None:
            lr = get_learning_rate(
                step=global_step,
                max_steps=max_steps,
                learning_rate=config.optimizer.learning_rate,
                warmup_steps=config.scheduler.warmup_steps,
                min_learning_rate=config.scheduler.min_learning_rate,
            )

            for param_group in optimizer.param_groups:
                param_group["lr"] = lr


        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        global_step += 1

        if (
            max_steps is not None
            and global_step >= max_steps
        ):
            break

    avg_loss = total_loss / num_batches

    return avg_loss, global_step