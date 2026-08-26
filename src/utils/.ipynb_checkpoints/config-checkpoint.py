from dataclasses import dataclass, field
import torch

@dataclass
class ModelConfig:
    
    vocab_size: int = 32000
    context_length : int = 1024

    d_model: int = 1024
    num_heads: int = 16
    num_layers: int = 24
    ff_hidden_dim: int = 4096

    dropout: float = 0.1
    eps : float = 1e-5
    bias: bool = True

    def __post_init__(self):
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads"
            )

        if self.ff_hidden_dim != 4 * self.d_model:
            raise ValueError(
                "GPT architecture expects ff_hidden_dim = 4 * d_model"
            )

        if self.context_length <= 0:
            raise ValueError(
                "context_length must be greater than 0"
            )

        if self.num_layers <= 0:
            raise ValueError(
                "num_layers must be greater than 0"
            )

        if self.num_heads <= 0:
            raise ValueError(
                "num_heads must be greater than 0"
            )

    @property
    def head_dim(self)-> int:
        return self.d_model // self.num_heads


@dataclass
class DataConfig:
    dataset_path : str = ("data/processed/fineweb_100k_clean.txt")
    context_length: int = 1024
    stride:int = 1024
    batch_size : int = 2

@dataclass
class OptimizerConfig:
    learning_rate : float = 3e-4
    weight_decay: float = 0.1

    beta1: float = 0.9
    beta2: float = 0.95

    adam_eps: float = 1e-8

@dataclass
class SchedulerConfig:
    warmup_steps: int = 500
    min_learning_rate: float = 3e-5
    max_steps: int = 30000

@dataclass
class TrainingConfig:

    max_grad_norm: float = 1.0

    epochs : int = 4
    device : torch.device  = field(
        default_factory = lambda: (
        torch.device("cuda" if torch.cuda.is_available() else "cpu"
                     )
        ))

    checkpoint_path: str = ("artifacts/gpt3large_checkpoints/best_checkpoint.pt")
    load_dir : str = "artifacts/tokenizer"

@dataclass
class GPTConfig:
    model : ModelConfig = field(
        default_factory=ModelConfig
    )

    data : DataConfig = field(
        default_factory= DataConfig
    )

    optimizer : OptimizerConfig = field(
        default_factory=OptimizerConfig
    )

    scheduler : SchedulerConfig = field(
        default_factory= SchedulerConfig
    )

    training : TrainingConfig = field(
        default_factory= TrainingConfig
    )


# ============================================================
# GPT-2 Baseline Configuration
# ============================================================

def gpt2_baseline_config() -> GPTConfig:

    return GPTConfig(
        model=ModelConfig(
            vocab_size=5000,
            d_model=256,
            num_heads=4,
            num_layers=6,
            ff_hidden_dim=1024,
            context_length=128,
        )
    )


# ============================================================
# GPT-3 Mini Configuration
# ============================================================

def gpt3_mini_config() -> GPTConfig:

    return GPTConfig(
        model=ModelConfig(
            vocab_size=5000,
            d_model=384,
            num_heads=6,
            num_layers=8,
            ff_hidden_dim=1536,
            context_length=256,
        )
    )


# ============================================================
# GPT-3 Small Configuration
# ============================================================

def gpt3_small_config() -> GPTConfig:

    return GPTConfig(
        model=ModelConfig(
            vocab_size=5000,
            d_model=512,
            num_heads=8,
            num_layers=10,
            ff_hidden_dim=2048,
            context_length=512,
        )
    )

def gpt3_large_config() -> GPTConfig:

    return GPTConfig(
        model=ModelConfig(
            vocab_size=5000,
            context_length=1024,
            d_model=1024,
            num_heads=16,
            num_layers=24,
            ff_hidden_dim=4096,
            dropout=0.1,
            eps=1e-5
        )
    )

GPT3_MODELS = {
    "gpt2_baseline": gpt2_baseline_config,
    "gpt3_mini": gpt3_mini_config,
    "gpt3_small": gpt3_small_config,
    "gpt3_large": gpt3_large_config
}