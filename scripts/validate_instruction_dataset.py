import json
import re
from pathlib import Path
from collections import Counter


INPUT_FILE = Path("data/instruction/raw/instruction_data_dolly.jsonl")
OUTPUT_FILE = Path("data/instruction/processed/clean.jsonl")


REQUIRED_FIELDS = {"instruction", "input", "response"}


def normalize_text(text: str) -> str:
    """
    Normalize whitespace while preserving the actual content.
    """

    # Convert non-breaking spaces to normal spaces
    text = text.replace("\u00a0", " ")

    # Normalize Windows / Unix line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove trailing spaces from each line
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def is_valid_example(example: dict) -> bool:
    """
    Validate a single instruction example.
    """

    if not isinstance(example, dict):
        return False

    # Required fields
    if not REQUIRED_FIELDS.issubset(example.keys()):
        return False

    instruction = example["instruction"]
    input_text = example["input"]
    response = example["response"]

    # Type validation
    if not isinstance(instruction, str):
        return False

    if not isinstance(input_text, str):
        return False

    if not isinstance(response, str):
        return False

    # Empty field validation
    if not instruction.strip():
        return False

    if not response.strip():
        return False

    return True


def create_dedup_key(example: dict) -> str:
    """
    Create a normalized key used for exact duplicate detection.
    """

    return (
        example["instruction"].lower().strip()
        + "\n"
        + example["input"].lower().strip()
        + "\n"
        + example["response"].lower().strip()
    )


def main():

    print("=" * 70)
    print("M1.2.3 — Dataset Validation & Normalization")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_FILE}"
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    valid = 0
    invalid = 0
    duplicates = 0

    invalid_json = 0
    missing_fields = 0
    invalid_types = 0
    empty_instruction = 0
    empty_response = 0

    seen = set()

    category_counts = Counter()

    instruction_lengths = []
    input_lengths = []
    response_lengths = []

    cleaned_examples = []

    print(f"\nInput : {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")

    with INPUT_FILE.open("r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, start=1):

            total += 1

            # --------------------------------------------------
            # JSON validation
            # --------------------------------------------------

            try:
                example = json.loads(line)
            except json.JSONDecodeError:
                invalid_json += 1
                invalid += 1
                continue

            # --------------------------------------------------
            # Structure validation
            # --------------------------------------------------

            if not isinstance(example, dict):
                invalid += 1
                continue

            if not REQUIRED_FIELDS.issubset(example.keys()):
                missing_fields += 1
                invalid += 1
                continue

            # --------------------------------------------------
            # Type validation
            # --------------------------------------------------

            if not all(
                isinstance(example[field], str)
                for field in ["instruction", "input", "response"]
            ):
                invalid_types += 1
                invalid += 1
                continue

            # --------------------------------------------------
            # Empty field validation
            # --------------------------------------------------

            if not example["instruction"].strip():
                empty_instruction += 1
                invalid += 1
                continue

            if not example["response"].strip():
                empty_response += 1
                invalid += 1
                continue

            # --------------------------------------------------
            # Normalization
            # --------------------------------------------------

            cleaned = {
                "instruction": normalize_text(example["instruction"]),
                "input": normalize_text(example["input"]),
                "response": normalize_text(example["response"]),
            }

            # --------------------------------------------------
            # Deduplication
            # --------------------------------------------------

            dedup_key = create_dedup_key(cleaned)

            if dedup_key in seen:
                duplicates += 1
                continue

            seen.add(dedup_key)

            # --------------------------------------------------
            # Statistics
            # --------------------------------------------------

            valid += 1

            category = example.get("category", "unknown")
            category_counts[category] += 1

            instruction_lengths.append(
                len(cleaned["instruction"])
            )

            input_lengths.append(
                len(cleaned["input"])
            )

            response_lengths.append(
                len(cleaned["response"])
            )

            cleaned_examples.append(cleaned)

    # ----------------------------------------------------------
    # Save cleaned dataset
    # ----------------------------------------------------------

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:

        for example in cleaned_examples:
            f.write(
                json.dumps(
                    example,
                    ensure_ascii=False
                )
                + "\n"
            )

    # ----------------------------------------------------------
    # Statistics helpers
    # ----------------------------------------------------------

    def stats(values):

        if not values:
            return {
                "min": 0,
                "max": 0,
                "mean": 0,
            }

        return {
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
        }

    instruction_stats = stats(instruction_lengths)
    input_stats = stats(input_lengths)
    response_stats = stats(response_lengths)

    # ----------------------------------------------------------
    # Print report
    # ----------------------------------------------------------

    print("\n" + "-" * 70)
    print("VALIDATION RESULTS")
    print("-" * 70)

    print(f"Total examples        : {total}")
    print(f"Valid examples        : {valid}")
    print(f"Invalid examples      : {invalid}")
    print(f"Duplicates removed    : {duplicates}")

    print("\nInvalid reasons:")
    print(f"  Invalid JSON         : {invalid_json}")
    print(f"  Missing fields       : {missing_fields}")
    print(f"  Invalid types        : {invalid_types}")
    print(f"  Empty instruction    : {empty_instruction}")
    print(f"  Empty response       : {empty_response}")

    print("\nCharacter statistics:")

    print(
        f"Instruction           : "
        f"min={instruction_stats['min']}, "
        f"max={instruction_stats['max']}, "
        f"mean={instruction_stats['mean']:.1f}"
    )

    print(
        f"Input                 : "
        f"min={input_stats['min']}, "
        f"max={input_stats['max']}, "
        f"mean={input_stats['mean']:.1f}"
    )

    print(
        f"Response              : "
        f"min={response_stats['min']}, "
        f"max={response_stats['max']}, "
        f"mean={response_stats['mean']:.1f}"
    )

    print("\nCategory distribution:")

    for category, count in category_counts.most_common():
        print(f"  {category:25s}: {count}")

    print("\nClean dataset saved:")
    print(f"  {OUTPUT_FILE}")

    print("\n" + "=" * 70)
    print("M1.2.3 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()