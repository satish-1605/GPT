import json
import random
from pathlib import Path

INPUT_FILE = Path("data/instruction/processed/clean.jsonl")

TRAIN_FILE =Path("data/instruction/processed/train.jsonl")
VAL_FILE  = Path("data/instruction/processed/val.jsonl")
TEST_FILE  = Path("data/instruction/processed/test.jsonl")

TRAIN_RATIO = 0.9
VAL_RATIO = 0.05
TEST_RATIO = 0.05

SEED = 42

def load_dataset():
    examples = []

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                example = json.loads(line)
                examples.append(example)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON at line {line_number}:{e}"
                )
    return examples

def save_dataset(examples, output_file):
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def verify_no_overlap(train, val, test):
    def make_keys(dataset):
        return {
            (
                example.get("instruction", ""),
                example.get("response", "")
            )
            for example in dataset
        }

    train_keys = make_keys(train)
    val_keys = make_keys(val)
    test_keys = make_keys(test)

    train_val = train_keys & val_keys
    val_test = val_keys & test_keys
    test_train = test_keys & train_keys

    if train_val or val_test or test_train:
        raise RuntimeError("Dataset overlap detected between splits")

    return True

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    print("=" * 70)
    print("INSTRUCTION DATASET SPLIT")
    print("=" * 70)

    dataset = load_dataset()

    total = len(dataset)

    if total == 0:
        raise ValueError("Dataset is empty")

    print(f"\n Total Examples: {total}")

    rng = random.Random(SEED)
    rng.shuffle(dataset)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train = dataset[:train_end]
    val = dataset[train_end:val_end]
    test = dataset[val_end:]

    assert len(train) + len(val) + len(test) == total

    verify_no_overlap(train, val, test)

    save_dataset(train, TRAIN_FILE)
    save_dataset(val, VAL_FILE)
    save_dataset(test, TEST_FILE)

    print("\nSplit configuration:")
    print(f"Seed: {SEED}")
    print(f"Train ratio: {TRAIN_RATIO:.0%}")
    print(f"Validation ratio: {VAL_RATIO:.0%}")
    print(f"Test ratio: {TEST_RATIO:.0%}")

    print("\nSplit results:")

    print(
        f"Train:      {len(train):>8} "
        f"({len(train) / total:.2%})"
    )

    print(
        f"Validation: {len(val):>8} "
        f"({len(val) / total:.2%})"
    )

    print(
        f"Test:       {len(test):>8} "
        f"({len(test) / total:.2%})"
    )

    print(
        f"Total:      {len(train) + len(val) + len(test):>8}"
    )

    print("\nOverlap check: PASSED")

    print("\nFiles created:")

    print(f"  {TRAIN_FILE}")
    print(f"  {VAL_FILE}")
    print(f"  {TEST_FILE}")

    print("\n" + "=" * 70)
    print("DATASET SPLIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()