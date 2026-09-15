import json
import torch
import random
from transformers import AutoTokenizer, GPT2LMHeadModel
from src.alignment.reward.reward_model import GPT2RewardModel
from pathlib import Path

MODEL_NAME = "gpt2-medium"

SFT_CHECKPOINT = "artifacts/sft/sft_best.pt"
REWARD_CHECKPOINT = "artifacts/reward/reward_best.pt"

TEST_FILE = Path("data/preference/processed/test.jsonl")

DEVICE = torch.device("cpu")
NUM_PROMPTS = 10

GENERATION_CONFIG = {
    "do_sample": True,
    "temperature": 0.9,
    "top_p": 0.95,
    "max_new_tokens": 128,
}

def format_prompt(instruction, input_text=""):

    if input_text:

        return (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Input:\n"
            f"{input_text}\n\n"
            "### Response:\n"
        )

    return (
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Response:\n"
    )

def load_prompts(file_path, num_prompts):

    prompts = []

    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
        num_prompts = min(num_prompts,len(data),)

        return random.sample(
        data,
        num_prompts,
    )

def load_sft_model():

    print("Loading SFT model...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(
        MODEL_NAME
    )

    checkpoint = torch.load(
        SFT_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    print("SFT model loaded successfully.")

    return model, tokenizer

def load_reward_model():
    model = GPT2RewardModel(
        model_name=MODEL_NAME,
        sft_checkpoint_path=SFT_CHECKPOINT,
    )

    checkpoint = torch.load(
        REWARD_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    print("Reward Model loaded successfully.")

    return model

@torch.no_grad()
def generate_response(model, tokenizer, prompt):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        **GENERATION_CONFIG,
    )

    generated_ids = output_ids[
        0,
        input_ids.shape[1]:,
    ]

    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    return response.strip()

@torch.no_grad()
def score_response(reward_model, tokenizer, prompt, response):
    text = (prompt + response + tokenizer.eos_token)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    reward = reward_model(input_ids, attention_mask)
    return reward.item()


def main():
    print(f"Device: {DEVICE}")
    print(f"Number of prompts: {NUM_PROMPTS}")

    sft_model, tokenizer = load_sft_model()

    reward_model = load_reward_model()
    prompts = load_prompts(TEST_FILE, NUM_PROMPTS)
    print(
        f"Loaded {len(prompts)} prompts."
    )

    for idx, item in enumerate(prompts, start=1):
        instruction = item["instruction"]
        input_text = item.get("input", "")

        prompt = format_prompt(
            instruction,
            input_text,
        )

        print("\n")
        print("=" * 80)
        print(f"PROMPT {idx}")
        print("=" * 80)
        print(prompt)

        response_a = generate_response(sft_model, tokenizer, prompt)
        response_b = generate_response(sft_model, tokenizer, prompt)

        reward_a = score_response(reward_model, tokenizer, prompt, response_a)
        reward_b = score_response(reward_model, tokenizer, prompt, response_b)

        print("-" * 80)
        print("RESPONSE A")
        print("-" * 80)

        print(response_a)

        print(
            f"\nReward A: {reward_a:.4f}"
        )

        print("-" * 80)
        print("RESPONSE B")
        print("-" * 80)

        print(response_b)

        print(
            f"\nReward B: {reward_b:.4f}"
        )

        print("-" * 80)

        if reward_a > reward_b:

            print(
                f"Reward Model preference: A "
                f"({reward_a:.4f} > {reward_b:.4f})"
            )

        elif reward_b > reward_a:

            print(
                f"Reward Model preference: B "
                f"({reward_b:.4f} > {reward_a:.4f})"
            )

        else:

            print("Reward Model preference: TIE")

if __name__ == "__main__":
    main()
