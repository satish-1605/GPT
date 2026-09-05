from dataclasses import dataclass
import torch


@dataclass
class SFTConfig:
    """ Configuration for Supervised Fine-Tuning (SFT). 
    Base model: Hugging Face GPT-2 Medium (~355M parameters) 
    This configuration controls only the SFT training process. 
    """
    model_name: str = "gpt2-medium"
    # ==========================================================
    # Training
    # ==========================================================

    epochs: int = 3

    learning_rate: float = 1e-5

    weight_decay: float = 0.1

    betas: tuple[float, float] = (0.9, 0.95)

    eps: float = 1e-8

    max_grad_norm: float = 1.0

    gradient_accumulation_steps: int = 1

    max_steps: int = 5


    # ==========================================================
    # Learning Rate Scheduler
    # ==========================================================

    scheduler: str = "cosine"

    warmup_ratio: float = 0.05

    min_lr_ratio: float = 0.1

    scheduler_max_steps: int = 5


    # ==========================================================
    # Data
    # ==========================================================

    batch_size: int = 2

    max_length: int = 1024

    num_workers: int = 0


    # ==========================================================
    # Evaluation / Logging
    # ==========================================================

    eval_interval: int = 10

    log_interval: int = 1


    # ==========================================================
    # Checkpointing
    # ==========================================================

    save_interval: int = 0

    save_best: bool = True


    # ==========================================================
    # Mixed Precision
    # ==========================================================

    use_amp: bool = True

    amp_dtype: str = "float16"


    # ==========================================================
    # Device
    # ==========================================================

    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    # ==========================================================
    # Reproducibility
    # ==========================================================

    seed: int = 42


    # ==========================================================
    # Dataset
    # ==========================================================

    train_file: str = (
        "data/instruction/processed/train.jsonl"
    )

    val_file: str = (
        "data/instruction/processed/val.jsonl"
    )

    test_file: str = (
        "data/instruction/processed/test.jsonl"
    )


    # ==========================================================
    # Checkpoints
    # ==========================================================

    checkpoint_dir: str = (
        "artifacts/sft"
    )