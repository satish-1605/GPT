import torch

from src.alignment.ppo.ppo_trainer import PPOTrainer

def test_ppo_one_training_step():
    model_name = "gpt2-medium"
    sft_checkpoint_path = "artifacts/sft/sft_best.pt"
    reward_checkpoint_path = "artifacts/reward/reward_best.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer = PPOTrainer(model_name=model_name,
                         sft_checkpoint_path=sft_checkpoint_path,
                         reward_checkpoint_path=reward_checkpoint_path,
                         device=device)

    prompts = [
    "Explain machine learning in simple terms.",
    "What is reinforcement learning?",
    "Explain recursion to a beginner.",
    "What is the purpose of PPO?",
    ]

    before = next(
        trainer.policy.parameters()
    ).detach().clone()

    metrics = trainer.train_step(
        prompts=prompts
    )

    required_metrics = [
        "policy_loss",
        "value_loss",
        "entropy",
        "rm_reward",
        "mean_kl",
    ]

    for key in required_metrics:

        assert key in metrics

        assert torch.isfinite(
            torch.tensor(metrics[key])
        ), f"{key} is NaN/Inf"

        after = next(
            trainer.policy.parameters()
        ).detach().clone()

        assert not torch.equal(
            before,
            after,
        ), "Policy parameters did not change"

        print("\nBatched PPO smoke test passed!")

        print(f"Device:      {device}")
        print(f"Batch size:  {len(prompts)}")
        print(f"Policy loss: {metrics['policy_loss']:.4f}")
        print(f"Value loss:  {metrics['value_loss']:.4f}")
        print(f"Entropy:     {metrics['entropy']:.4f}")
        print(f"RM reward:   {metrics['rm_reward']:.4f}")
        print(f"Mean KL:     {metrics['mean_kl']:.4f}")