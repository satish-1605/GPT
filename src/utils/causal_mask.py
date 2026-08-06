import torch

def create_causal_mask(seq_len: int, device=None):
    """
    Creates a causal (lower-triangular) attention mask.

    Args:
        seq_len (int): Sequence length.
        device: Torch device.

    Returns:
        Tensor of shape (1, 1, seq_len, seq_len)
        True = allowed attention
        False = masked (future positions)
    """
    mask = torch.tril(
        torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
    )
    return mask.unsqueeze(0).unsqueeze(0)

