import torch.nn as Module
from torch.optim import AdamW

def create_optimizer(
        model: Module,
        learning_rate : float,
        weight_decay : float = 0.01,
        betas : tuple[float, float] = (0.9, 0.999),
        eps : float = 1e-8,
        )-> AdamW:
    
    return AdamW(
        params = model.parameters(),
        lr = learning_rate,
        betas = betas,
        eps = eps,
        weight_decay = weight_decay,
    )

