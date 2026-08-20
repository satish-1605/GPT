from dataclasses import dataclass
import torch

@dataclass
class GPTConfig:
    
    # model configurations
    vocab_size: int = 5000
    d_model: int = 256
    num_heads: int = 4
    num_layers: int = 6
    ff_hidden_dim: int = 1024
    dropout: float = 0.1
    eps : float = 1e-5
    bias: bool = True

    # Dataset configurations
    fineweb_dataset_path : str = "data/processed/fineweb_10k_clean.txt"
    context_length: int = 128
    stride:int = 128
    batch_size : int = 16

    # Optimizer Configuration
    learning_rate : float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    adam_eps: float = 1e-8

    # Learning-Rate Schedule
    warmup_steps: int = 100
    min_learning_rate: float = 3e-5
    max_steps: int = 5000

    # Gradient Configuration
    max_grad_norm: float = 1.0

    # Training Configuration
    epochs : int = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # artifacts path
    checkpoint_path: str = "artifacts/checkpoints/best_checkpoint.pt"
    load_dir : str = "artifacts/tokenizer"

    
    def __post_init__(self):
        assert self.d_model % self.num_heads == 0, (
            "d_model must be divisible by num_heads"
        )