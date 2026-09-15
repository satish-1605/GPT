import json
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from src.alignment.reward.reward_model import GPT2RewardModel


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "gpt2-medium"

SFT_CHECKPOINT_PATH = (
    "artifacts/sft/sft_best.pt"
)

PPO_CHECKPOINT_PATH = (
    "artifacts/ppo/ppo_final.pt"
)

REWARD_CHECKPOINT_PATH = (
    "artifacts/reward/reward_best.pt"
)

EVAL_DATA_PATH = (
    "data/rlhf/eval_prompts.jsonl"
)

SFT_OUTPUT_PATH = (
    "artifacts/ppo/sft_baseline.jsonl"
)

PPO_OUTPUT_PATH = (
    "artifacts/ppo/ppo_evaluation.jsonl"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_NEW_TOKENS = 50
TEMPERATURE = 1.0
TOP_P = 0.9


# ============================================================
# Load prompts
# ============================================================

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


# ============================================================
# Load SFT policy
# ============================================================

def load_sft_model(
    model_name,
    checkpoint_path,
    device,
):

    model = GPT2LMHeadModel.from_pretrained(
        model_name
    )

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

    return model


# ============================================================
# Load PPO policy
# ============================================================

def load_ppo_model(
    model_name,
    checkpoint_path,
    device,
):

    model = GPT2LMHeadModel.from_pretrained(
        model_name
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["policy_state_dict"]
    )

    model.to(device)
    model.eval()

    return model


# ============================================================
# Load Reward Model
# ============================================================

def load_reward_model(
    model_name,
    checkpoint_path,
    device,
):

    model = GPT2RewardModel(
        model_name,
        SFT_CHECKPOINT_PATH,
    )

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

    for param in model.parameters():
        param.requires_grad = False

    return model


# ============================================================
# Generate response
# ============================================================

@torch.no_grad()
def generate_response(
    model,
    tokenizer,
    prompt,
    device,
):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=(
            model.config.n_positions
            - MAX_NEW_TOKENS
        ),
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        pad_token_id=tokenizer.eos_token_id,
    )

    response_ids = output_ids[
        :,
        input_ids.shape[1]:,
    ]

    response = tokenizer.decode(
        response_ids[0],
        skip_special_tokens=True,
    ).strip()

    return response, output_ids, attention_mask


# ============================================================
# Score response with Reward Model
# ============================================================

@torch.no_grad()
def score_response(
    reward_model,
    tokenizer,
    output_ids,
    device,
):

    attention_mask = torch.ones_like(
        output_ids,
        device=device,
    )

    reward = reward_model(
        input_ids=output_ids,
        attention_mask=attention_mask,
    )

    reward = reward.squeeze().item()

    return reward


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("SFT + PPO EVALUATION")
    print("=" * 70)

    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Evaluation data: {EVAL_DATA_PATH}")
    print()

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    tokenizer = GPT2Tokenizer.from_pretrained(
        MODEL_NAME
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    print("Loading SFT model...")

    sft_model = load_sft_model(
        MODEL_NAME,
        SFT_CHECKPOINT_PATH,
        DEVICE,
    )

    print("Loading PPO model...")

    ppo_model = load_ppo_model(
        MODEL_NAME,
        PPO_CHECKPOINT_PATH,
        DEVICE,
    )

    print("Loading Reward Model...")

    reward_model = load_reward_model(
        MODEL_NAME,
        REWARD_CHECKPOINT_PATH,
        DEVICE,
    )

    # --------------------------------------------------------
    # Prompts
    # --------------------------------------------------------

    prompts = load_prompts(
        EVAL_DATA_PATH
    )

    print(f"Evaluation prompts: {len(prompts)}")
    print()

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    Path(SFT_OUTPUT_PATH).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # SFT BASELINE EVALUATION
    # ========================================================

    print("=" * 70)
    print("SFT BASELINE EVALUATION")
    print("=" * 70)

    sft_results = []

    total_sft_reward = 0.0

    for index, prompt in enumerate(prompts):

        response, output_ids, _ = generate_response(
            model=sft_model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=DEVICE,
        )

        reward = score_response(
            reward_model=reward_model,
            tokenizer=tokenizer,
            output_ids=output_ids,
            device=DEVICE,
        )

        total_sft_reward += reward

        result = {
            "index": index,
            "prompt": prompt,
            "response": response,
            "reward": reward,
        }

        sft_results.append(result)

        if (
            (index + 1) % 10 == 0
            or index == 0
        ):

            print(
                f"[{index + 1:3d}/{len(prompts)}] "
                f"Reward: {reward:.4f}"
            )

    # --------------------------------------------------------
    # Save SFT results
    # --------------------------------------------------------

    with open(
        SFT_OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        for result in sft_results:

            f.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

    mean_sft_reward = (
        total_sft_reward / len(sft_results)
    )

    print()
    print("=" * 70)
    print("SFT BASELINE COMPLETE")
    print("=" * 70)
    print(f"Prompts evaluated: {len(sft_results)}")
    print(f"Mean RM reward: {mean_sft_reward:.4f}")
    print(f"Results: {SFT_OUTPUT_PATH}")

    # ========================================================
    # PPO EVALUATION
    # ========================================================

    print()
    print("=" * 70)
    print("PPO EVALUATION")
    print("=" * 70)

    ppo_results = []

    total_ppo_reward = 0.0

    for index, prompt in enumerate(prompts):

        response, output_ids, _ = generate_response(
            model=ppo_model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=DEVICE,
        )

        reward = score_response(
            reward_model=reward_model,
            tokenizer=tokenizer,
            output_ids=output_ids,
            device=DEVICE,
        )

        total_ppo_reward += reward

        result = {
            "index": index,
            "prompt": prompt,
            "response": response,
            "reward": reward,
        }

        ppo_results.append(result)

        if (
            (index + 1) % 10 == 0
            or index == 0
        ):

            print(
                f"[{index + 1:3d}/{len(prompts)}] "
                f"Reward: {reward:.4f}"
            )

    # --------------------------------------------------------
    # Save PPO results
    # --------------------------------------------------------

    Path(PPO_OUTPUT_PATH).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        PPO_OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        for result in ppo_results:

            f.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

    mean_ppo_reward = (
        total_ppo_reward / len(ppo_results)
    )

    # ========================================================
    # Final comparison
    # ========================================================

    reward_delta = (
        mean_ppo_reward
        - mean_sft_reward
    )

    if mean_sft_reward != 0:
        reward_improvement_pct = (
            reward_delta
            / abs(mean_sft_reward)
            * 100
        )
    else:
        reward_improvement_pct = 0.0

    print()
    print("=" * 70)
    print("PPO EVALUATION COMPLETE")
    print("=" * 70)

    print(f"Prompts evaluated: {len(ppo_results)}")
    print(f"SFT mean RM reward: {mean_sft_reward:.4f}")
    print(f"PPO mean RM reward: {mean_ppo_reward:.4f}")
    print(f"Reward delta: {reward_delta:+.4f}")
    print(
        f"Reward change: "
        f"{reward_improvement_pct:+.2f}%"
    )

    print(f"SFT results: {SFT_OUTPUT_PATH}")
    print(f"PPO results: {PPO_OUTPUT_PATH}")


if __name__ == "__main__":
    main()