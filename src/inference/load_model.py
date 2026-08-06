from src.utils.checkpoint import load_checkpoint
import torch

def load_model(model, filepath, device):
    """
    Load a trained GPT model for inference.

    Args:
        model: GPT model instance.
        checkpoint_path: Path to checkpoint file.
        device: cpu or cuda.

    Returns:
        model: Loaded model in evaluation mode.
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)
    model.eval()

    return model