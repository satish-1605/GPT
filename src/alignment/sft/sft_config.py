from dataclasses import dataclass
import torch

@dataclass
class SFTConfig:
    """
    Configuration for Supervised Fine-Tuning (SFT).

    The model architecture remains the same as the
    pretrained 300M Base GPT. This configuration only
    controls the SFT training process.
    """

    # Training
    epochs: int = 3
    learning_rate: float = 1e-5
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1


    # Learning-rate scheduler
    scheduler: str = "cosine"
    warmup_ratio: float = 0.05
    min_lr_ratio: float = 0.1

    # Data
    batch_size: int = 2
    max_length: int = 1024
    num_workers: int = 0
    max_steps: int =10


    # Evaluation
    eval_interval: int = 500
    log_interval: int = 100

    # Checkpointing
    save_interval: int = 1000
    save_best: bool = True

    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "float16"

    #device 
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
                         
    
    # Reproducibility
    seed: int = 42

    # Paths
    train_file: str = (
        "data/instruction/processed/train.jsonl"
    )

    val_file: str = (
        "data/instruction/processed/val.jsonl"
    )

    test_file: str = (
        "data/instruction/processed/test.jsonl"
    )

    checkpoint_dir: str = (
        "artifacts/sft"
    )

    base_checkpoint: str = (
        "artifacts/gpt3large_checkpoints/best_checkpoint.pt"
    )

    tokenizer_path: str = (
        "artifacts/tokenizer/tokenizer.json"
    )

        