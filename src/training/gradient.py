import torch
from torch.nn import Module
from torch.nn.utils import clip_grad_norm_


def clip_gradients(
    model: Module,
    max_grad_norm: float,
) -> float:

    total_norm = clip_grad_norm_(
        model.parameters(),
        max_norm=max_grad_norm,
    )

    return total_norm.item()