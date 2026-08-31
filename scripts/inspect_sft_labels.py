from pathlib import Path

from src.alignment.sft.instruction_dataset import InstructionDataset
from src.alignment.sft.sft_config import SFTConfig
from src.tokenizer.hf_tokenizer import HFTokenizer


def main():

    config = SFTConfig()

    tokenizer = HFTokenizer(
        config.tokenizer_path
    )

    dataset = InstructionDataset(
        Path(config.train_file),
        tokenizer,
        config.max_length
    )

    example = dataset[0]

    input_ids = example["input_ids"]
    labels = example["labels"]

    print("=" * 70)
    print("SFT LABEL SANITY CHECK")
    print("=" * 70)

    print(f"Sequence length : {len(input_ids)}")

    print(
        f"Input tokens    : "
        f"{sum(x != tokenizer.pad_token_id for x in input_ids)}"
    )

    print(
        f"Masked labels   : "
        f"{sum(x == -100 for x in labels)}"
    )

    print(
        f"Active labels   : "
        f"{sum(x != -100 for x in labels)}"
    )

    print("\nFirst 30 labels:")
    print(labels[:30].tolist())

    print("\nFirst active labels:")
    active_labels = [
        x.item()
        for x in labels
        if x.item() != -100
    ]

    print(active_labels[:30])

    print("\nDecoded active labels:")

    decoded = tokenizer.decode(
        active_labels
    )

    print(decoded)

    print("=" * 70)


if __name__ == "__main__":
    main()