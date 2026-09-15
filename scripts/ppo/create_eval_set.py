import json
import random
from pathlib import Path


INPUT_PATH = Path("data/rlhf/ppo_prompts.jsonl")
EVAL_PATH = Path("data/rlhf/eval_prompts.jsonl")
PPO_PATH = Path("data/rlhf/ppo_prompts.jsonl")

EVAL_SIZE = 100
SEED = 42


def load_prompts(path):
    prompts = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            example = json.loads(line)
            prompt = example["prompt"].strip()

            if prompt:
                prompts.append(prompt)

    return prompts


def save_prompts(prompts, path):
    with open(path, "w", encoding="utf-8") as f:
        for prompt in prompts:
            f.write(
                json.dumps(
                    {"prompt": prompt},
                    ensure_ascii=False,
                )
                + "\n"
            )


def main():
    prompts = load_prompts(INPUT_PATH)

    print(f"Original prompts: {len(prompts)}")

    # Remove duplicate prompts
    unique_prompts = list(dict.fromkeys(prompts))

    print(f"Unique prompts: {len(unique_prompts)}")
    print(f"Duplicates removed: {len(prompts) - len(unique_prompts)}")

    if len(unique_prompts) <= EVAL_SIZE:
        raise ValueError(
            "Not enough unique prompts to create evaluation set."
        )

    # Reproducible sampling
    random.seed(SEED)

    eval_prompts = random.sample(
        unique_prompts,
        EVAL_SIZE,
    )

    eval_set = set(eval_prompts)

    # Remaining prompts become PPO training data
    train_prompts = [
        prompt
        for prompt in unique_prompts
        if prompt not in eval_set
    ]

    # Safety checks
    assert len(eval_prompts) == EVAL_SIZE
    assert len(eval_set) == EVAL_SIZE
    assert len(eval_set.intersection(train_prompts)) == 0
    assert len(train_prompts) + len(eval_prompts) == len(unique_prompts)

    save_prompts(
        eval_prompts,
        EVAL_PATH,
    )

    save_prompts(
        train_prompts,
        PPO_PATH,
    )

    print()
    print("Evaluation split created successfully")
    print(f"Evaluation prompts: {len(eval_prompts)}")
    print(f"PPO training prompts: {len(train_prompts)}")
    print(f"Overlap: {len(eval_set.intersection(train_prompts))}")
    print(f"Evaluation file: {EVAL_PATH}")
    print(f"PPO file: {PPO_PATH}")


if __name__ == "__main__":
    main()