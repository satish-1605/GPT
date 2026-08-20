import torch.nn as Module
from torch.optim import AdamW

def create_optimizer(
        model: Module,
        learning_rate : float,
        weight_decay : float ,
        betas : tuple[float, float],
        eps : float,
        )-> AdamW:
    
    return AdamW(
        params = model.parameters(),
        lr = learning_rate,
        betas = betas,
        eps = eps,
        weight_decay = weight_decay,
    )

