from pathlib import Path
import torch
from src.training.checkpoint import load_checkpoint

def resume_training(
        checkpoint_path : str | Path,
        model,
        optimizer,
        device):
     """
    Restore model, optimizer, and training state
    from a checkpoint.
    """

     (epoch, global_step, best_val_loss) = load_checkpoint(path=checkpoint_path, 
                                                           model=model, 
                                                           optimizer=optimizer, 
                                                           device= device)

     return {
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
    }