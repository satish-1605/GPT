import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.alignment.reward.reward_config import RewardTrainingConfig
from src.alignment.reward.reward_model import GPT2RewardModel
from src.alignment.reward.preference_dataset import PreferenceDataset, preference_collate_fn
from src.alignment.reward.reward_loss import preference_loss

def evaluate(model, dataloader, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_pairs = 0

    with torch.no_grad():
        for batch in dataloader:
            chosen_input_ids = batch["chosen_input_ids"].to(device)
            chosen_attention_mask = batch["chosen_attention_mask"].to(device)

            rejected_input_ids = batch["rejected_input_ids"].to(device)
            rejected_attention_mask = batch["rejected_attention_mask"].to(device)

            chosen_rewards = model(chosen_input_ids, chosen_attention_mask)
            rejected_rewards = model(rejected_input_ids, rejected_attention_mask)

            loss = preference_loss(chosen_rewards, rejected_rewards)

            total_loss += loss.item() * chosen_rewards.size(0)

            correct = (chosen_rewards > rejected_rewards).sum().item()
            total_correct += correct

            total_pairs += chosen_rewards.size(0)

    avg_loss = total_loss / total_pairs
    accuracy = total_correct / total_pairs

    return avg_loss, accuracy



def main():
    config = RewardTrainingConfig()

    device = torch.device(
        config.device
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer loaded: {config.model_name}")

    model = GPT2RewardModel(model_name=config.model_name,
                            sft_checkpoint_path=config.sft_checkpoint_path,)

    model.to(device)
    print("Reward Model loaded successfully.")

    train_dataset = PreferenceDataset(file_path=config.train_file, tokenizer=tokenizer,
        max_length=config.max_length,)

    val_dataset = PreferenceDataset(file_path=config.val_file, tokenizer=tokenizer,
        max_length=config.max_length,)

    test_dataset = PreferenceDataset(file_path=config.test_file, tokenizer=tokenizer,
        max_length=config.max_length,)

    print(f"Train pairs: {len(train_dataset)}")
    print(f"Val pairs:   {len(val_dataset)}")
    print(f"Test pairs:  {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
        shuffle=True, collate_fn=lambda batch: preference_collate_fn(
            batch,
            tokenizer.pad_token_id,
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda batch: preference_collate_fn(
            batch,
            tokenizer.pad_token_id,
        ),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda batch: preference_collate_fn(
            batch,
            tokenizer.pad_token_id,
        ),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    import os

    os.makedirs(
        config.checkpoint_dir,
        exist_ok=True,
    )

    best_val_loss = float("inf")

    for epoch in range(config.epochs):
        model.train()

        total_train_loss = 0.0
        total_correct = 0
        total_pairs = 0
        for batch_idx, batch in enumerate(train_loader):

            chosen_input_ids = batch[
                "chosen_input_ids"
            ].to(device)

            chosen_attention_mask = batch[
                "chosen_attention_mask"
            ].to(device)

            rejected_input_ids = batch[
                "rejected_input_ids"
            ].to(device)

            rejected_attention_mask = batch[
                "rejected_attention_mask"
            ].to(device)

            # --------------------------------------------------
            # Forward pass
            # --------------------------------------------------

            chosen_rewards = model(
                chosen_input_ids,
                chosen_attention_mask,
            )

            rejected_rewards = model(
                rejected_input_ids,
                rejected_attention_mask,
            )

            # --------------------------------------------------
            # Bradley-Terry loss
            # --------------------------------------------------

            loss = preference_loss(
                chosen_rewards,
                rejected_rewards,
            )

            # --------------------------------------------------
            # Backpropagation
            # --------------------------------------------------

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            # --------------------------------------------------
            # Statistics
            # --------------------------------------------------

            batch_size = chosen_rewards.size(0)

            total_train_loss += (
                loss.item() * batch_size
            )

            correct = (
                chosen_rewards > rejected_rewards
            ).sum().item()

            total_correct += correct
            total_pairs += batch_size

            if (batch_idx + 1) % 50 == 0:

                print(
                    f"Epoch [{epoch + 1}/{config.epochs}] "
                    f"Batch [{batch_idx + 1}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        # --------------------------------------------------
        # Epoch statistics
        # --------------------------------------------------

        train_loss = (
            total_train_loss / total_pairs
        )

        train_accuracy = (
            total_correct / total_pairs
        )

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            device,
        )

        print(
            f"\nEpoch [{epoch + 1}/{config.epochs}]"
        )

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Train Accuracy: "
            f"{train_accuracy:.4f}"
        )

        print(
            f"Val Loss: {val_loss:.4f}"
        )

        print(
            f"Val Accuracy: "
            f"{val_accuracy:.4f}"
        )

        # --------------------------------------------------
        # Save best model
        # --------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            checkpoint_path = (
                f"{config.checkpoint_dir}/"
                "reward_best.pt"
            )

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                },
                checkpoint_path,
            )

            print(
                f"Best Reward Model saved to "
                f"{checkpoint_path}"
            )

    # --------------------------------------------------
    # Test
    # --------------------------------------------------

    print("\nRunning final test evaluation...")

    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        device,
    )

    print(
        f"Test Loss: {test_loss:.4f}"
    )

    print(
        f"Test Accuracy: {test_accuracy:.4f}"
    )

    print("\nReward Model training completed.")


if __name__ == "__main__":
    main()

