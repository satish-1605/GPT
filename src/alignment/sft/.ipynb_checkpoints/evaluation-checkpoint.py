import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, GPT2LMHeadModel

from src.alignment.sft.sft_config import SFTConfig
from src.alignment.sft.instruction_dataset import InstructionDataset


def main():

    config = SFTConfig()

    device = torch.device(config.device)

    checkpoint_path = (
        Path(config.checkpoint_dir)
        / "sft_best.pt"
    )

    print("=" * 70)
    print("SFT TEST EVALUATION")
    print("=" * 70)
    print(f"Device     : {device}")
    print(f"Model      : {config.model_name}")
    print(f"Checkpoint : {checkpoint_path}")
    print("=" * 70)

    # --------------------------------------------------
    # Tokenizer
    # --------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name
    )

    tokenizer.pad_token = tokenizer.eos_token

    # --------------------------------------------------
    # Test dataset
    # --------------------------------------------------

    print("\nLoading test dataset...")

    test_dataset = InstructionDataset(
        config.test_file,
        tokenizer,
        config.max_length,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Test examples : {len(test_dataset):,}")
    print(f"Test batches  : {len(test_loader):,}")

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    print("\nLoading base model...")

    model = GPT2LMHeadModel.from_pretrained(
        config.model_name
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    # --------------------------------------------------
    # Load SFT checkpoint
    # --------------------------------------------------

    print("\nLoading SFT checkpoint...")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"Checkpoint step    : "
        f"{checkpoint['global_step']:,}"
    )

    print(
        f"Best validation loss: "
        f"{checkpoint['best_val_loss']:.4f}"
    )

    model.to(device)
    model.eval()

    # --------------------------------------------------
    # Loss
    # --------------------------------------------------

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=-100
    )

    total_loss = 0.0
    num_batches = 0

    print("\nEvaluating test set...")

    with torch.no_grad():

        for batch in test_loader:

            input_ids = batch["input_ids"].to(
                device,
                non_blocking=True,
            )

            attention_mask = batch["attention_mask"].to(
                device,
                non_blocking=True,
            )

            labels = batch["labels"].to(
                device,
                non_blocking=True,
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                logits = outputs.logits

                shifted_logits = logits[:, :-1, :]
                shifted_labels = labels[:, 1:]

                loss = loss_fn(
                    shifted_logits.reshape(
                        -1,
                        shifted_logits.size(-1),
                    ),
                    shifted_labels.reshape(-1),
                )

            total_loss += loss.item()
            num_batches += 1

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    test_loss = total_loss / num_batches
    test_perplexity = math.exp(test_loss)

    print()
    print("=" * 70)
    print("SFT TEST RESULTS")
    print("=" * 70)
    print(f"Test Loss       : {test_loss:.4f}")
    print(f"Test Perplexity : {test_perplexity:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()