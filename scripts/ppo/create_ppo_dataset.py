import json
from pathlib import Path


SOURCE_PATH = Path("data/instruction/processed/clean.jsonl")
OUTPUT_PATH = Path("data/rlhf/ppo_prompts.jsonl")


def build_prompt(instruction, input_text):
    instruction = instruction.strip()
    input_text = input_text.strip()

    if input_text:
        return f"{instruction}\n\n{input_text}"

    return instruction


def main():

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    count = 0

    with SOURCE_PATH.open(
        "r",
        encoding="utf-8",
    ) as source, OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as output:

        for line in source:

            if not line.strip():
                continue

            example = json.loads(line)

            instruction = example.get(
                "instruction",
                "",
            )

            input_text = example.get(
                "input",
                "",
            )

            prompt = build_prompt(
                instruction,
                input_text,
            )

            if not prompt:
                continue

            output.write(
                json.dumps(
                    {"prompt": prompt},
                    ensure_ascii=False,
                )
                + "\n"
            )

            count += 1

    print(f"Created PPO dataset: {OUTPUT_PATH}")
    print(f"Total prompts: {count}")


if __name__ == "__main__":
    main()