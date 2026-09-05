from src.alignment.sft.instruction_dataset import InstructionDataset
from src.alignment.sft.sft_config import SFTConfig
from src.tokenizer.hf_tokenizer import HFTokenizer


def main():

    config = SFTConfig()

    tokenizer = HFTokenizer(
        config.tokenizer_path
    )

    dataset = InstructionDataset(
        config.train_file,
        tokenizer,
        config.max_length,
    )

    example = dataset.examples[0]

    print("=" * 70)
    print("RAW EXAMPLE")
    print("=" * 70)

    print(example)

    prompt, response = dataset.format_instruction(
        example
    )

    print()
    print("=" * 70)
    print("PROMPT")
    print("=" * 70)

    print(prompt)

    print()
    print("=" * 70)
    print("RESPONSE")
    print("=" * 70)

    print(response)

    token_ids, labels = dataset._encode(
        prompt,
        response
    )

    print()
    print("=" * 70)
    print("TOKEN INFORMATION")
    print("=" * 70)

    print("Prompt tokens :", len(
        tokenizer.encode(prompt)
    ))

    print("Response tokens :", len(
        tokenizer.encode(response)
    ))

    print("Total tokens :", len(token_ids))

    print(
        "First 30 input IDs:",
        token_ids[:30]
    )

    print(
        "First 30 labels:",
        labels[:30]
    )

    print()
    print("=" * 70)
    print("DECODED INPUT")
    print("=" * 70)

    print(
        tokenizer.decode(token_ids)
    )

    print()
    print("=" * 70)
    print("TARGET RESPONSE TOKENS")
    print("=" * 70)

    response_token_ids = [
        token_id
        for token_id, label in zip(
            token_ids,
            labels
        )
        if label != -100
    ]

    print(
        tokenizer.decode(response_token_ids)
    )


if __name__ == "__main__":
    main()