import torch

def compute_perplexity(loss: float):
    """
    Compute perplexity from cross-entropy loss.

    Args:
        loss (float): Average cross-entropy loss.

    Returns:
        float: Perplexity.
    """

    return torch.exp(torch.tensor(loss)).item()