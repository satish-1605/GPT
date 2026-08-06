import torch
from src.losses.loss import calculate_loss


def validate(loader, model, device):
    total_loss = 0.0
    model.eval()

    with torch.no_grad():
        for input_ids, target_ids in loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            logits = model(input_ids)
            loss = calculate_loss(logits, target_ids)    
    
            total_loss += loss.item()
    val_loss = total_loss/len(loader)
    return val_loss 