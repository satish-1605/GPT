import torch
import json
from pathlib import Path
from transformers import AutoTokenizer, GPT2LMHeadModel

from src.alignment.sft.sft_config import SFTConfig


INPUT_FILE = Path(
    "data/preference/raw/preference_prompts_200.jsonl"
)

OUTPUT_FILE = Path(
    "data/preference/raw/candidates.jsonl"
)

NUM_CANDIDATES = 4

GENERATION_CONFIG = {
    "do_sample": True,
    "temperature": 0.9,
    "top_p": 0.95,
    "max_new_tokens": 256,
}


def format_prompt(instruction, input=""):
    if input:
        prompt = (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Input:\n"
            f"{input}\n\n"
            "### Response:\n"
        )
    else:
        prompt = (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Response:\n"
        )

    return prompt


def generate_response(
    model,
    tokenizer,
    prompt,
    generation_config,
):
    model.eval()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        output = model.generate(
            **inputs,
            **generation_config,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Remove prompt tokens
    generated_tokens = output[
        0
    ][inputs["input_ids"].shape[1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return response.strip()


def main():

    # Load SFT configuration
    config = SFTConfig()

    device = config.device

    checkpoint_path = (
        Path(config.checkpoint_dir)
        / "sft_best.pt"
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name
    )

    tokenizer.pad_token = tokenizer.eos_token

    # Load base model architecture
    model = GPT2LMHeadModel.from_pretrained(
        config.model_name
    )

    model.config.pad_token_id = (
        tokenizer.pad_token_id
    )

    # Load SFT checkpoint
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    print(f"Loaded SFT model from: {checkpoint_path}")
    print(f"Using device: {device}")

    # Create output directory
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Load prompts
    with INPUT_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        prompts = [
            json.loads(line)
            for line in f
        ]

    print(f"Loaded {len(prompts)} prompts")

    # Generate candidates
    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        for idx, item in enumerate(
            prompts,
            start=1,
        ):

            instruction = item["instruction"]

            input_text = item.get(
                "input",
                "",
            )

            prompt = format_prompt(
                instruction,
                input_text,
            )

            candidates = []

            for cand_idx in range(
                NUM_CANDIDATES
            ):

                response = generate_response(
                    model,
                    tokenizer,
                    prompt,
                    GENERATION_CONFIG,
                )

                candidates.append({
                    "candidate_id": chr(
                        65 + cand_idx
                    ),
                    "response": response,
                })

            output_item = {
                "id": idx,
                "instruction": instruction,
                "input": input_text,
                "candidates": candidates,
            }

            f.write(
                json.dumps(
                    output_item,
                    ensure_ascii=False,
                ) + "\n"
            )

            print(
                f"Generated "
                f"{NUM_CANDIDATES} candidates "
                f"for prompt "
                f"{idx}/{len(prompts)}"
            )

    print(
        "Candidate generation completed successfully."
    )


if __name__ == "__main__":
    main()