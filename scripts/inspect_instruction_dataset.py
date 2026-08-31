import json
import statistics
from pathlib import Path


INPUT_FILE = Path("data/instruction/processed/clean.jsonl")
REPORT_FILE = Path("data/instruction/processed/statistics.txt")

NUM_SAMPLES = 5


def load_dataset():
    examples = []

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Invalid JSON at line {line_number}: {e}")

    return examples


def get_length_stats(values):
    if not values:
        return {
            "count": 0,
            "min": 0,
            "max": 0,
            "mean": 0,
            "median": 0,
        }

    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
    }


def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    examples = load_dataset()

    print("=" * 70)
    print("INSTRUCTION DATASET INSPECTION")
    print("=" * 70)

    total = len(examples)

    print(f"\nTotal examples: {total}")

    if total == 0:
        print("\nDataset is empty.")
        return

    instructions = []
    responses = []

    empty_instructions = 0
    empty_responses = 0

    for example in examples:

        instruction = str(example.get("instruction", "")).strip()
        response = str(example.get("response", "")).strip()

        if not instruction:
            empty_instructions += 1

        if not response:
            empty_responses += 1

        instructions.append(instruction)
        responses.append(response)

    # Character lengths
    instruction_lengths = [
        len(x) for x in instructions if x
    ]

    response_lengths = [
        len(x) for x in responses if x
    ]

    instruction_stats = get_length_stats(instruction_lengths)
    response_stats = get_length_stats(response_lengths)

    # Duplicate detection
    pairs = [
        (instruction, response)
        for instruction, response in zip(instructions, responses)
    ]

    unique_pairs = set(pairs)
    duplicate_count = total - len(unique_pairs)

    # Report
    report_lines = []

    report_lines.append("INSTRUCTION DATASET STATISTICS")
    report_lines.append("=" * 70)

    report_lines.append(f"Total examples: {total}")
    report_lines.append(
        f"Unique instruction-response pairs: {len(unique_pairs)}"
    )
    report_lines.append(
        f"Duplicate examples: {duplicate_count}"
    )

    report_lines.append("")
    report_lines.append("EMPTY VALUES")
    report_lines.append("-" * 70)
    report_lines.append(
        f"Empty instructions: {empty_instructions}"
    )
    report_lines.append(
        f"Empty responses: {empty_responses}"
    )

    report_lines.append("")
    report_lines.append("INSTRUCTION LENGTH (characters)")
    report_lines.append("-" * 70)

    for key, value in instruction_stats.items():
        report_lines.append(f"{key}: {value}")

    report_lines.append("")
    report_lines.append("RESPONSE LENGTH (characters)")
    report_lines.append("-" * 70)

    for key, value in response_stats.items():
        report_lines.append(f"{key}: {value}")

    report_lines.append("")
    report_lines.append("SAMPLE EXAMPLES")
    report_lines.append("=" * 70)

    for i, example in enumerate(examples[:NUM_SAMPLES], start=1):

        report_lines.append(f"\nExample {i}")
        report_lines.append("-" * 70)
        report_lines.append(
            f"Instruction: {example.get('instruction', '')}"
        )
        report_lines.append(
            f"Response: {example.get('response', '')}"
        )

    report = "\n".join(report_lines)

    print("\n" + report)

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with REPORT_FILE.open("w", encoding="utf-8") as f:
        f.write(report)

    print("\n" + "=" * 70)
    print(f"Statistics saved to: {REPORT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()