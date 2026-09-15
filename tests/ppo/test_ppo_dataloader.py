from scripts.ppo.ppo_dataloader import (
    create_ppo_dataloader,
)


def test_ppo_dataloader():

    dataloader = create_ppo_dataloader(
        data_path="data/rlhf/ppo_prompts.jsonl",
        batch_size=4,
        max_samples=10,
        shuffle=False,
    )

    batch = next(iter(dataloader))

    assert "prompt" in batch
    assert len(batch["prompt"]) == 4

    for prompt in batch["prompt"]:
        assert isinstance(prompt, str)
        assert prompt.strip() != ""

    print("\nPPO DataLoader smoke test passed!")
    print(f"Batch size: {len(batch['prompt'])}")

    for i, prompt in enumerate(batch["prompt"]):
        print(f"\nPrompt {i + 1}:")
        print(prompt)


if __name__ == "__main__":
    test_ppo_dataloader()