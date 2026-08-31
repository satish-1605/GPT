from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.alignment.sft.sft_config import SFTConfig
from src.alignment.sft.instruction_dataset import InstructionDataset
from src.tokenizer.hf_tokenizer import HFTokenizer
from src.models.gpt import GPT
from src.utils.config import GPTConfig
from src.alignment.sft.sft_trainer import SFTTrainer


def main():

    config = SFTConfig()

    device = torch.device(config.device)

    # ==================================================
    # Tokenizer
    # ==================================================

    tokenizer = HFTokenizer(
        config.tokenizer_path
    )

    # ==================================================
    # Dataset
    # ==================================================

    train_dataset = InstructionDataset(
        Path(config.train_file),
        tokenizer,
        config.max_length
    )

    val_dataset = InstructionDataset(
        Path(config.val_file),
        tokenizer,
        config.max_length
    )

    # ==================================================
    # DataLoader
    # ==================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )

    # ==================================================
    # Model
    # ==================================================

    gpt_config = GPTConfig()

    model = GPT(
        gpt_config
    ).to(device)

    # ==================================================
    # Load Base GPT
    # ==================================================

    base_checkpoint = torch.load(
        config.base_checkpoint,
        map_location=device,
        weights_only=False
    )

    # Your HF portable checkpoint contains the state
    # dictionary directly.
    if "model_state_dict" in base_checkpoint:

        model.load_state_dict(
            base_checkpoint["model_state_dict"]
        )

    else:

        model.load_state_dict(
            base_checkpoint
        )

    print(
        "Base GPT checkpoint loaded successfully."
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
    # Resume SFT
    # ==================================================

    resume_path = (
        Path(config.checkpoint_dir)
        / "sft_best.pt"
    )

    if resume_path.exists():

        checkpoint = torch.load(
            resume_path,
            map_location=device,
            weights_only=False
        )

        # ----------------------------------------------
        # Restore model
        # ----------------------------------------------

        trainer.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        # ----------------------------------------------
        # Restore optimizer
        # ----------------------------------------------

        trainer.optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        # ----------------------------------------------
        # Restore AMP scaler
        # ----------------------------------------------

        if (
            trainer.use_amp
            and "scaler_state_dict" in checkpoint
        ):
            trainer.scaler.load_state_dict(
                checkpoint["scaler_state_dict"]
            )

        # ----------------------------------------------
        # Restore training state
        # ----------------------------------------------

        trainer.global_step = (
            checkpoint["global_step"]
        )

        trainer.best_val_loss = (
            checkpoint.get(
                "best_val_loss",
                float("inf")
            )
        )

        # ----------------------------------------------
        # IMPORTANT:
        # Do NOT restore old scheduler state.
        #
        # We created a fresh scheduler using:
        # scheduler_max_steps = 6000
        # ----------------------------------------------

        print(
            f"Resumed SFT from step "
            f"{trainer.global_step:,}"
        )

        print(
            f"Best validation loss: "
            f"{trainer.best_val_loss:.4f}"
        )

    else:

        print(
            "No SFT checkpoint found. "
            "Starting SFT from base GPT."
        )

    # ==================================================
    # Train
    # ==================================================

    trainer.train()


if __name__ == "__main__":
    main()