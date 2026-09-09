from pathlib import Path

class RewardTrainingConfig:

    # Model
    model_name = "gpt2-medium"
    sft_checkpoint_path = "artifacts/sft/sft_best.pt"

    # Dataset
    train_file = Path("data/preference/processed/train.jsonl")
    val_file = Path("data/preference/processed/validation.jsonl")
    test_file = Path("data/preference/processed/test.jsonl")

    # Tokenization
    max_length = 1024

    # Training
    batch_size = 2
    learning_rate = 1e-5
    epochs = 3

    # Optimization
    weight_decay = 0.01

    # Checkpointing
    checkpoint_dir = "artifacts/reward"

    # Device
    device = "cuda"