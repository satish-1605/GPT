import torch


def save_checkpoint(epoch, model, optimizer, train_loss, val_loss, 
                    filepath: str = "checkpoint.pt"):
    
    checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    "train_loss": train_loss,
    "val_loss": val_loss,
    }

    torch.save(checkpoint, filepath)

def load_checkpoint(model, optimizer=None, device="cpu", filepath: str = "checkpoint.pt"):
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss']
    val_loss = checkpoint['val_loss']

    return epoch, train_loss, val_loss


