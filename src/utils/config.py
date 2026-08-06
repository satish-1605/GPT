from dataclasses import dataclass
import torch

@dataclass
class GPTConfig:
    # Vocabulary 
    vocab_size: int = 5000

    #load dir for tokenizer
    load_dir : str = "artifacts/tokenizer"
    max_stories : int = 1000

    # Sequence
    max_seq_len: int = 128

    # Transformer dimensions
    d_model: int = 256
    num_heads: int = 4
    num_layers: int = 6
    ff_hidden_dim: int = 1024  # 4 * d_model

    # Regularization
    dropout: float = 0.1

    # layer norm
    eps : float = 1e-5

    # Optional (commonly used)
    bias: bool = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #training params
    batch_size : int = 16
    learning_rate : float = 3e-4
    epochs : int = 10
    checkpoint_path : str = "checkpoint.pt"


    def __post_init__(self):
        assert self.d_model % self.num_heads == 0, (
            "d_model must be divisible by num_heads"
        )