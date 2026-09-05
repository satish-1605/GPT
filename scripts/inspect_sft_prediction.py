import torch

from src.alignment.sft.sft_config import SFTConfig
from src.alignment.sft.instruction_dataset import InstructionDataset
from src.tokenizer.hf_tokenizer import HFTokenizer
from src.models.gpt import GPT
from src.utils.config import GPTConfig


def main():

    print("=" * 70)
    print("SFT PREDICTION DIAGNOSTIC")
    print("=" * 70)

    sft_config = SFTConfig()
    gpt_config = GPTConfig()

    tokenizer = HFTokenizer(
        sft_config.tokenizer_path
    )

    dataset = InstructionDataset(
        sft_config.train_file,
        tokenizer,
        sft_config.max_length
    )

    model = GPT(gpt_config).to(
        sft_config.device
    )

    checkpoint = torch.load(
        "artifacts/sft/sft_best.pt",
        map_location=sft_config.device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # --------------------------------------------------
    # First training example
    # --------------------------------------------------

    example = dataset.examples[0]

    instruction = example["instruction"]
    input_text = example.get("input", "")
    expected = example["response"]

    if input_text:
        prompt = (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Input:\n"
            f"{input_text}\n\n"
            "### Response:\n"
        )
    else:
        prompt = (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Response:\n"
        )

    response = expected + "<|endoftext|>"

    prompt_ids = tokenizer.encode(prompt)
    response_ids = tokenizer.encode(response)

    input_ids = torch.tensor(
        [prompt_ids + response_ids],
        dtype=torch.long,
        device=sft_config.device
    )

    labels = torch.tensor(
        [[-100] * len(prompt_ids) + response_ids],
        dtype=torch.long,
        device=sft_config.device
    )

    print("\nINSTRUCTION:")
    print(instruction)

    print("\nEXPECTED RESPONSE:")
    print(expected)

    print("\nPROMPT TOKEN COUNT:")
    print(len(prompt_ids))

    print("\nRESPONSE TOKEN COUNT:")
    print(len(response_ids))

    print("\nPROMPT:")
    print(repr(prompt))

    print("\nDECODED PROMPT:")
    print(
        tokenizer.decode(prompt_ids)
    )

    # --------------------------------------------------
    # Forward pass
    # --------------------------------------------------

    with torch.no_grad():

        logits = model(
            input_ids
        )

        # Next-token prediction
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]

        loss_fn = torch.nn.CrossEntropyLoss(
            ignore_index=-100
        )

        loss = loss_fn(
            shift_logits.reshape(
                -1,
                shift_logits.size(-1)
            ),
            shift_labels.reshape(-1)
        )

        predictions = torch.argmax(
            shift_logits,
            dim=-1
        )

    print("\nTEACHER-FORCED LOSS:")
    print(f"{loss.item():.6f}")

    # --------------------------------------------------
    # Compare predicted vs expected response tokens
    # --------------------------------------------------

    active_positions = (
        shift_labels[0] != -100
    )

    predicted_ids = predictions[0][
        active_positions
    ].tolist()

    expected_ids = shift_labels[0][
        active_positions
    ].tolist()

    print("\nEXPECTED TOKEN IDS:")
    print(expected_ids)

    print("\nPREDICTED TOKEN IDS:")
    print(predicted_ids)

    print("\nEXPECTED DECODED:")
    print(
        tokenizer.decode(expected_ids)
    )

    print("\nPREDICTED DECODED:")
    print(
        tokenizer.decode(predicted_ids)
    )

    correct = sum(
        p == e
        for p, e in zip(
            predicted_ids,
            expected_ids
        )
    )

    total = len(expected_ids)

    print("\nTOKEN ACCURACY:")
    print(
        f"{correct}/{total} "
        f"= {correct / total:.2%}"
    )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()