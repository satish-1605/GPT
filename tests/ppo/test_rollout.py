import torch

from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer

from src.alignment.ppo.value_model import ValueModel
from src.alignment.ppo.rollout import Rollout


def test_batched_rollout():

    model_name = "gpt2-medium"

    sft_checkpoint_path = (
        "artifacts/sft/sft_best.pt"
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    tokenizer = GPT2Tokenizer.from_pretrained(
        model_name
    )

    tokenizer.pad_token = tokenizer.eos_token

    checkpoint = torch.load(
        sft_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    sft_state_dict = checkpoint[
        "model_state_dict"
    ]

    policy = GPT2LMHeadModel.from_pretrained(
        model_name
    )

    policy.load_state_dict(
        sft_state_dict
    )

    policy.to(device)
    policy.eval()

    reference_model = GPT2LMHeadModel.from_pretrained(
        model_name
    )

    reference_model.load_state_dict(
        sft_state_dict
    )

    reference_model.to(device)
    reference_model.eval()

    value_model = ValueModel(
        model_name=model_name,
        sft_checkpoint_path=sft_checkpoint_path,
    )

    value_model.to(device)
    value_model.eval()

    rollout = Rollout(
        policy=policy,
        reference_model=reference_model,
        value_model=value_model,
        tokenizer=tokenizer,
        device=device,
    )

    prompts = [
        "Explain machine learning simply.",
        "What is reinforcement learning?",
        "Explain recursion to a beginner.",
        "What is the purpose of PPO?",
    ]

    data = rollout.generate(
        prompts=prompts,
        max_new_tokens=20,
    )

    # ---------------------------------------------
    # Basic checks
    # ---------------------------------------------

    batch_size = len(prompts)

    assert data["input_ids"].shape[0] == batch_size
    assert data["response_ids"].shape[0] == batch_size

    assert (
        data["old_log_probs"].shape
        == data["response_ids"].shape
    )

    assert (
        data["ref_log_probs"].shape
        == data["response_ids"].shape
    )

    assert (
        data["values"].shape
        == data["response_ids"].shape
    )

    assert (
        data["response_mask"].shape
        == data["response_ids"].shape
    )

    # Check numerical stability

    for key in [
        "old_log_probs",
        "ref_log_probs",
        "values",
    ]:
        assert torch.isfinite(
            data[key]
        ).all(), f"{key} contains NaN/Inf"

    print("\nBatched rollout smoke test passed!")
    print(f"Device: {device}")
    print(
        f"Response shape: "
        f"{data['response_ids'].shape}"
    )


if __name__ == "__main__":
    test_batched_rollout()