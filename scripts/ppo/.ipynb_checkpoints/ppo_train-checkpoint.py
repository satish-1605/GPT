import os
import torch

from scripts.ppo.ppo_dataloader import create_ppo_dataloader
from src.alignment.ppo.ppo_trainer import PPOTrainer


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "gpt2-medium"

SFT_CHECKPOINT_PATH = "artifacts/sft/sft_best.pt"
REWARD_CHECKPOINT_PATH = "artifacts/reward/reward_best.pt"

PPO_DATA_PATH = "data/rlhf/ppo_prompts.jsonl"

OUTPUT_DIR = "artifacts/ppo"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

FINAL_CHECKPOINT_PATH = os.path.join(
    OUTPUT_DIR,
    "ppo_final.pt",
)

MAX_SAMPLES = 1000
BATCH_SIZE = 1
EPOCHS = 1

LOG_EVERY = 50
SAVE_EVERY = 250

MAX_NEW_TOKENS = 50


# ============================================================
# PPO hyperparameters
# ============================================================

CLIP_EPSILON = 0.2
GAMMA = 0.99
GAE_LAMBDA = 0.95

KL_COEF = 0.1
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01


# ============================================================
# Checkpoint helper
# ============================================================

def save_checkpoint(
    trainer,
    global_step,
    checkpoint_path,
):
    checkpoint = {
    "policy_state_dict": trainer.policy.state_dict(),
    "value_model_state_dict": trainer.value_model.state_dict(),
    "global_step": global_step,
    }
    
    torch.save(
        checkpoint,
        FINAL_CHECKPOINT_PATH,
    )

    print(
        f"Checkpoint saved: {checkpoint_path}"
    )


# ============================================================
# Main
# ============================================================

def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("PPO TRAINING")
    print("=" * 70)

    print(f"Device: {device}")
    print(f"Model: {MODEL_NAME}")
    print(f"Samples: {MAX_SAMPLES}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Log every: {LOG_EVERY} steps")
    print(f"Save every: {SAVE_EVERY} steps")
    print()

    # --------------------------------------------------------
    # Create output directories
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    dataloader = create_ppo_dataloader(
        data_path=PPO_DATA_PATH,
        batch_size=BATCH_SIZE,
        max_samples=MAX_SAMPLES,
        shuffle=True,
    )

    steps_per_epoch = len(dataloader)
    total_steps = steps_per_epoch * EPOCHS

    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total PPO steps: {total_steps}")
    print()

    # --------------------------------------------------------
    # PPO Trainer
    # --------------------------------------------------------

    trainer = PPOTrainer(
        model_name=MODEL_NAME,
        sft_checkpoint_path=SFT_CHECKPOINT_PATH,
        reward_checkpoint_path=REWARD_CHECKPOINT_PATH,
        device=device,
        clip_epsilon=CLIP_EPSILON,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        kl_coef=KL_COEF,
        value_coef=VALUE_COEF,
        entropy_coef=ENTROPY_COEF,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    global_step = 0

    # ========================================================
    # Training loop
    # ========================================================

    for epoch in range(EPOCHS):

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS}"
        )

        for batch_idx, batch in enumerate(dataloader):

            prompts = batch["prompt"]

            metrics = trainer.train_step(
                prompts=prompts,
            )

            global_step += 1

            # ------------------------------------------------
            # Logging
            # ------------------------------------------------

            if (
                global_step % LOG_EVERY == 0
                or global_step == 1
                or global_step == total_steps
            ):

                policy_lr = (
                    trainer.policy_optimizer
                    .param_groups[0]["lr"]
                )

                value_lr = (
                    trainer.value_optimizer
                    .param_groups[0]["lr"]
                )

                print(
                    f"Step {global_step:4d}/{total_steps} | "
                    f"Epoch {epoch + 1}/{EPOCHS} | "
                    f"Policy Loss: {metrics['policy_loss']:.4f} | "
                    f"Value Loss: {metrics['value_loss']:.4f} | "
                    f"Entropy: {metrics['entropy']:.4f} | "
                    f"RM Reward: {metrics['rm_reward']:.4f} | "
                    f"mean_log_ratio: "
                    f"{metrics['mean_log_ratio']:.4f} | "
                    f"Policy LR: {policy_lr:.2e} | "
                    f"Value LR: {value_lr:.2e}"
                )

            # ------------------------------------------------
            # Periodic checkpoint
            # ------------------------------------------------

            if (
                global_step % SAVE_EVERY == 0
                and global_step != total_steps
            ):

                checkpoint_path = os.path.join(
                    CHECKPOINT_DIR,
                    f"ppo_step_{global_step}.pt",
                )

                save_checkpoint(
                    trainer=trainer,
                    global_step=global_step,
                    checkpoint_path=checkpoint_path,
                )

    # ========================================================
    # Save final checkpoint
    # ========================================================

    save_checkpoint(
        trainer=trainer,
        global_step=global_step,
        checkpoint_path=FINAL_CHECKPOINT_PATH,
    )

    print()
    print("=" * 70)
    print("PPO TRAINING COMPLETE")
    print("=" * 70)
    print(f"Final checkpoint: {FINAL_CHECKPOINT_PATH}")
    print(f"Global steps: {global_step}")

    print()
    print("Periodic checkpoints:")

    for step in range(
        SAVE_EVERY,
        total_steps,
        SAVE_EVERY,
    ):
        checkpoint_path = os.path.join(
            CHECKPOINT_DIR,
            f"ppo_step_{step}.pt",
        )
        print(checkpoint_path)


if __name__ == "__main__":
    main()