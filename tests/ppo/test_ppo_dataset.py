from scripts.ppo.ppo_dataset import PPODataset

def test_ppo_dataset():
    dataset = PPODataset(
        data_path="data/rlhf/ppo_prompts.jsonl",
        max_samples=10,
    )

    assert len(dataset) == 10

    sample = dataset[0]

    assert "prompt" in sample
    assert isinstance(
        sample["prompt"],
        str,
    )

    assert sample["prompt"].strip() != ""

    print("\nPPO dataset smoke test passed!")
    print(f"Dataset size: {len(dataset)}")
    print(f"First prompt: {sample['prompt']}")

if __name__ == "__main__":
    test_ppo_dataset()