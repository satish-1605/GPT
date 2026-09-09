from pathlib import Path
import json


TOKENIZER_DIR = Path("artifacts/tokenizer")


def main():

    tokenizer_path = TOKENIZER_DIR / "tokenizer.json"
    merges_path = TOKENIZER_DIR / "merges.txt"

    with tokenizer_path.open("r", encoding="utf-8") as f:
        tokenizer = json.load(f)

    # Hugging Face BPE stores merges in model.merges
    merges = tokenizer["model"]["merges"]

    with merges_path.open("w", encoding="utf-8") as f:

        for merge in merges:

            # Newer tokenizers may store merges as "a b"
            if isinstance(merge, str):
                f.write(merge + "\n")

            # Older format may be ["a", "b"]
            else:
                f.write(
                    f"{merge[0]} {merge[1]}\n"
                )

    print(f"Saved {len(merges):,} merges")
    print(f"Path: {merges_path.resolve()}")


if __name__ == "__main__":
    main()