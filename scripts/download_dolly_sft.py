from pathlib import Path
import json

from datasets import load_dataset

DATASET_NAME = "databricks/databricks-dolly-15k"

OUTPUT_DIR = Path("data/instruction/raw")
OUTPUT_FILE = OUTPUT_DIR / "instruction_data_dolly.jsonl"

def main():

    print("=" * 60)
    print("M1.2.2 — Instruction Dataset Acquisition")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nDataset : {DATASET_NAME}")
    print("Downloading dataset...")

    dataset = load_dataset(DATASET_NAME, split="train")

    print("\n Download complete.")
    print(f"Number of examples: {len(dataset)}")

    print("\n Dataset Features")
    print(dataset.features)

    print(f"\n Saving to:")
    print(OUTPUT_FILE)

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for example in dataset:
            record = {
                "instruction": example['instruction'],
                "input":  example["context"] or "",
                "response": example['response'],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\n Saved Succesfully")

    with OUTPUT_FILE.open("r", encoding="utf-8") as f:
        line_count = sum(1 for _ in f)

    print(f"Saved example: {line_count}")
    print(f"Output file size: {OUTPUT_FILE.stat().st_size / (1024 * 1024):.2f} MB")

    print("\n First Example : ")
    with OUTPUT_FILE.open("r", encoding="utf-8") as f:
        first = json.loads(f.readline())

    print(json.dumps(first, indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)
    print("M1.2.2 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()