import json
import random
from pathlib import Path

import numpy as np
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

OUTPUT_PATH = (
    "artifacts/ppo/controlled_sft_vs_ppo.jsonl"
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

MAX_NEW_TOKENS = 50
TEMPERATURE = 1.0
TOP_P = 0.9

BASE_SEED = 1000


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Data
# ============================================================

def load_prompts(path):
    prompts = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            if not line.strip():
                continue

            example = json.loads(line)

            prompt = example["prompt"].strip()

            if prompt:
                prompts.append(prompt)

    return prompts


# ============================================================
# Model loading
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


def load_reward_model(
    model_name,
    checkpoint_path,
    sft_checkpoint_path,
    device,
):
    model = GPT2RewardModel(
        model_name,
        sft_checkpoint_path,
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
# Generation
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

    attention_mask = inputs[
        "attention_mask"
    ].to(device)

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

    return response, output_ids


# ============================================================
# Reward
# ============================================================

@torch.no_grad()
def score_response(
    reward_model,
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
    print("CONTROLLED SFT vs PPO EVALUATION")
    print("=" * 70)

    print(f"Device       : {DEVICE}")
    print(f"Model        : {MODEL_NAME}")
    print(f"Evaluation   : {EVAL_DATA_PATH}")
    print(f"Base seed    : {BASE_SEED}")
    print(
        f"Temperature  : {TEMPERATURE}"
    )
    print(
        f"Top-p        : {TOP_P}"
    )
    print(
        f"Max tokens   : {MAX_NEW_TOKENS}"
    )

    print()

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    tokenizer = GPT2Tokenizer.from_pretrained(
        MODEL_NAME
    )

    tokenizer.pad_token = (
        tokenizer.eos_token
    )

    tokenizer.padding_side = "left"

    # --------------------------------------------------------
    # Load models
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
        SFT_CHECKPOINT_PATH,
        DEVICE,
    )

    # --------------------------------------------------------
    # Load evaluation prompts
    # --------------------------------------------------------

    prompts = load_prompts(
        EVAL_DATA_PATH
    )

    print()
    print(
        f"Evaluation prompts: {len(prompts)}"
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    results = []

    total_sft_reward = 0.0
    total_ppo_reward = 0.0

    improved = 0
    worsened = 0
    unchanged = 0

    for index, prompt in enumerate(prompts):

        seed = BASE_SEED + index

        # ====================================================
        # SFT
        # ====================================================

        set_seed(seed)

        sft_response, sft_output_ids = (
            generate_response(
                model=sft_model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=DEVICE,
            )
        )

        sft_reward = score_response(
            reward_model=reward_model,
            output_ids=sft_output_ids,
            device=DEVICE,
        )

        # ====================================================
        # PPO
        # ====================================================

        # Reset to EXACT same seed.
        set_seed(seed)

        ppo_response, ppo_output_ids = (
            generate_response(
                model=ppo_model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=DEVICE,
            )
        )

        ppo_reward = score_response(
            reward_model=reward_model,
            output_ids=ppo_output_ids,
            device=DEVICE,
        )

        # ====================================================
        # Comparison
        # ====================================================

        reward_delta = (
            ppo_reward
            - sft_reward
        )

        total_sft_reward += sft_reward
        total_ppo_reward += ppo_reward

        if reward_delta > 0.05:
            improved += 1

        elif reward_delta < -0.05:
            worsened += 1

        else:
            unchanged += 1

        result = {
            "index": index,
            "seed": seed,
            "prompt": prompt,
            "sft_response": sft_response,
            "ppo_response": ppo_response,
            "sft_reward": sft_reward,
            "ppo_reward": ppo_reward,
            "reward_delta": reward_delta,
        }

        results.append(result)

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            index == 0
            or (index + 1) % 10 == 0
        ):
            print(
                f"[{index + 1:3d}/{len(prompts)}] "
                f"SFT={sft_reward:+.4f} | "
                f"PPO={ppo_reward:+.4f} | "
                f"Δ={reward_delta:+.4f}"
            )

    # --------------------------------------------------------
    # Aggregate metrics
    # --------------------------------------------------------

    num_prompts = len(results)

    mean_sft_reward = (
        total_sft_reward
        / num_prompts
    )

    mean_ppo_reward = (
        total_ppo_reward
        / num_prompts
    )

    mean_delta = (
        mean_ppo_reward
        - mean_sft_reward
    )

    reward_change_pct = 0.0

    if mean_sft_reward != 0:
        reward_change_pct = (
            mean_delta
            / abs(mean_sft_reward)
            * 100
        )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    Path(OUTPUT_PATH).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        for result in results:

            f.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()

    print("=" * 70)
    print("CONTROLLED EVALUATION COMPLETE")
    print("=" * 70)

    print(
        f"Prompts compared       : {num_prompts}"
    )

    print(
        f"Mean SFT RM reward     : "
        f"{mean_sft_reward:.4f}"
    )

    print(
        f"Mean PPO RM reward     : "
        f"{mean_ppo_reward:.4f}"
    )

    print(
        f"Mean reward delta      : "
        f"{mean_delta:+.4f}"
    )

    print(
        f"Reward change          : "
        f"{reward_change_pct:+.2f}%"
    )

    print()

    print(
        f"PPO improved           : "
        f"{improved}"
    )

    print(
        f"PPO worsened           : "
        f"{worsened}"
    )

    print(
        f"Approximately same     : "
        f"{unchanged}"
    )

    print()

    print(
        f"Results saved to       : "
        f"{OUTPUT_PATH}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()