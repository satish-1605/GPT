from pathlib import Path

import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

from src.alignment.sft.sft_config import SFTConfig


def load_model(model_name, checkpoint_path=None, device="cpu"):
    model = GPT2LMHeadModel.from_pretrained(model_name)

    if checkpoint_path is not None:
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


def generate_response(
    model,
    tokenizer,
    prompt,
    device,
):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():

        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][input_ids.shape[1]:]

    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    return response.strip()


def main():

    config = SFTConfig()

    device = torch.device(config.device)

    checkpoint_path = (
        Path(config.checkpoint_dir)
        / "sft_best.pt"
    )

    print("=" * 70)
    print("SFT GENERATION EVALUATION")
    print("=" * 70)
    print(f"Device     : {device}")
    print(f"Model      : {config.model_name}")
    print(f"SFT Checkpoint : {checkpoint_path}")
    print("=" * 70)

    # --------------------------------------------------
    # Tokenizer
    # --------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name
    )

    tokenizer.pad_token = tokenizer.eos_token

    # --------------------------------------------------
    # Load base model
    # --------------------------------------------------

    print("\nLoading BASE GPT-2 Medium...")

    base_model = load_model(
        config.model_name,
        checkpoint_path=None,
        device=device,
    )

    # --------------------------------------------------
    # Load SFT model
    # --------------------------------------------------

    print("\nLoading SFT GPT-2 Medium...")

    sft_model = load_model(
        config.model_name,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    # --------------------------------------------------
    # Evaluation prompts
    # --------------------------------------------------

    prompts = [
        "### Instruction:\nExplain what machine learning is in simple terms.\n\n### Response:\n",

        "### Instruction:\nWrite a short Python function to calculate the factorial of a number.\n\n### Response:\n",

        "### Instruction:\nWhat are the main benefits of using renewable energy?\n\n### Response:\n",

        "### Instruction:\nExplain the difference between supervised and unsupervised learning.\n\n### Response:\n",

        "### Instruction:\nWrite a professional email asking for a meeting with your manager.\n\n### Response:\n",

        "### Instruction:\nGive me three tips for improving my programming skills.\n\n### Response:\n",

        "### Instruction:\nWhat is the purpose of a neural network?\n\n### Response:\n",

        "### Instruction:\nExplain recursion with a simple example.\n\n### Response:\n",
    ]

    # --------------------------------------------------
    # Generate
    # --------------------------------------------------

    print("\n")
    print("=" * 70)
    print("GENERATION COMPARISON")
    print("=" * 70)

    for i, prompt in enumerate(prompts, start=1):

        print(f"\n{'-' * 70}")
        print(f"Example {i}")
        print(f"{'-' * 70}")

        print("\nINSTRUCTION:")
        print(prompt.split("### Response:")[0])

        print("\nBASE GPT-2:")
        base_response = generate_response(
            base_model,
            tokenizer,
            prompt,
            device,
        )
        print(base_response)

        print("\nSFT GPT-2:")
        sft_response = generate_response(
            sft_model,
            tokenizer,
            prompt,
            device,
        )
        print(sft_response)

    print("\n")
    print("=" * 70)
    print("GENERATION EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()