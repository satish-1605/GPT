import torch
from src.losses.loss import calculate_loss


def validate(loader, model, device):
    
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for input_ids, target_ids in loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            logits = model(input_ids)
            loss = calculate_loss(logits, target_ids)    
    
            total_loss += loss.item()
            num_batches += 1
    val_loss = total_loss/num_batches
    return val_loss 