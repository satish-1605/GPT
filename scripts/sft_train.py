import torch

from torch.utils.data import DataLoader

from transformers import (
AutoTokenizer,
GPT2LMHeadModel,
)

from src.alignment.sft.sft_config import SFTConfig
from src.alignment.sft.instruction_dataset import InstructionDataset
from src.alignment.sft.sft_trainer import SFTTrainer

def main():


    # ==================================================
    # Configuration
    # ==================================================

    config = SFTConfig()

    device = torch.device(
        config.device
    )

    print("=" * 70)
    print("SFT TRAINING")
    print("=" * 70)

    print(
        f"Device        : {device}"
    )

    print(
        f"Model         : {config.model_name}"
    )

    print(
        f"Max steps     : "
        f"{config.max_steps:,}"
    )

    print(
        f"Batch size    : "
        f"{config.batch_size}"
    )

    print(
        f"Learning rate : "
        f"{config.learning_rate}"
    )

    print(
        f"Max length    : "
        f"{config.max_length}"
    )

    print("=" * 70)

    # ==================================================
    # Tokenizer
    # ==================================================

    print()
    print("Loading Hugging Face tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name
    )
    tokenizer.model_max_length = config.max_length

    # GPT-2 does not define a PAD token.
    # Use EOS as PAD.
    tokenizer.pad_token = tokenizer.eos_token

    print(
        f"Tokenizer vocab size: "
        f"{len(tokenizer):,}"
    )

    print(
        f"EOS token ID: "
        f"{tokenizer.eos_token_id}"
    )

    print(
        f"PAD token ID: "
        f"{tokenizer.pad_token_id}"
    )

    # ==================================================
    # Dataset
    # ==================================================

    print()
    print("Loading datasets...")

    train_dataset = InstructionDataset(
        config.train_file,
        tokenizer,
        config.max_length,
    )

    val_dataset = InstructionDataset(
        config.val_file,
        tokenizer,
        config.max_length,
    )

    print(
        f"Train examples: "
        f"{len(train_dataset):,}"
    )

    print(
        f"Val examples  : "
        f"{len(val_dataset):,}"
    )

    # ==================================================
    # DataLoader
    # ==================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    print()
    print(
        f"Train batches : "
        f"{len(train_loader):,}"
    )

    print(
        f"Val batches   : "
        f"{len(val_loader):,}"
    )

    # ==================================================
    # Model
    # ==================================================

    print()
    print(
        f"Loading Hugging Face model: "
        f"{config.model_name}"
    )

    model = GPT2LMHeadModel.from_pretrained(
        config.model_name
    )

    # Match model padding configuration
    model.config.pad_token_id = (
        tokenizer.pad_token_id
    )

    model.to(device)

    # ==================================================
    # Parameter Count
    # ==================================================

    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Total parameters    : "
        f"{total_params:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_params:,}"
    )

    # ==================================================
    # Trainer
    # ==================================================

    trainer = SFTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )

    # ==================================================
    # Fresh SFT Run
    # ==================================================

    print()
    print("=" * 70)
    print("STARTING FRESH SFT RUN")
    print("=" * 70)

    print(
        "No previous SFT checkpoint "
        "will be resumed."
    )

    # ==================================================
    # Train
    # ==================================================

    trainer.train()


if __name__ == "__main__":
    main()
