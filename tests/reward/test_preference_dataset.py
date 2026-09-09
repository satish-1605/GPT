import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from pathlib import Path

from src.alignment.reward.preference_dataset import PreferenceDataset, preference_collate_fn

def main():
    tokenizer = AutoTokenizer.from_pretrained(
        "gpt2-medium"
    )
    tokenizer.pad_token = tokenizer.eos_token

    train_file = Path("data/preference/processed/train.jsonl")

    dataset = PreferenceDataset(train_file, tokenizer)

    print(f"Dataset size: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=4, shuffle=True,
                        collate_fn=lambda batch : preference_collate_fn(
                            batch, tokenizer.pad_token_id))

    batch = next(iter(loader))

    chosen_input_ids = batch["chosen_input_ids"]
    chosen_attention_mask = batch["chosen_attention_mask"]

    rejected_input_ids = batch["rejected_input_ids"]
    rejected_attention_mask = batch["rejected_attention_mask"]

    print(
        f"Chosen input shape: "
        f"{chosen_input_ids.shape}"
    )

    print(
        f"Chosen attention mask shape: "
        f"{chosen_attention_mask.shape}"
    )

    print(
        f"Rejected input shape: "
        f"{rejected_input_ids.shape}"
    )

    print(
        f"Rejected attention mask shape: "
        f"{rejected_attention_mask.shape}"
    )

    assert chosen_input_ids.dtype == torch.long
    assert rejected_input_ids.dtype == torch.long

    assert chosen_attention_mask.dtype == torch.long
    assert rejected_attention_mask.dtype == torch.long

    assert chosen_input_ids.ndim == 2
    assert rejected_input_ids.ndim == 2

    assert chosen_attention_mask.shape == chosen_input_ids.shape
    assert rejected_attention_mask.shape == rejected_input_ids.shape

    # Batch size
    assert chosen_input_ids.shape[0] == 4
    assert rejected_input_ids.shape[0] == 4

    # Attention mask contains only 0 and 1
    assert torch.all(
        (chosen_attention_mask == 0)
        | (chosen_attention_mask == 1)
    )

    assert torch.all(
        (rejected_attention_mask == 0)
        | (rejected_attention_mask == 1)
    )
    chosen_padding = (
        chosen_input_ids == tokenizer.pad_token_id
    )

    rejected_padding = (
        rejected_input_ids == tokenizer.pad_token_id
    )

    assert torch.all(
        chosen_attention_mask[chosen_padding] == 0
    )

    assert torch.all(
        rejected_attention_mask[rejected_padding] == 0
    )

    print("\nAll Dataset + DataLoader tests passed.")


if __name__ == "__main__":
    main()