from pathlib import Path

from src.tokenizer.hf_tokenizer import HFTokenizer


TOKENIZER_PATH = Path(
    "artifacts/tokenizer/tokenizer.json"
)


def main():

    print("=" * 60)
    print("HF Tokenizer Test")
    print("=" * 60)

    tokenizer = HFTokenizer(
        TOKENIZER_PATH
    )

    print(
        f"PAD token ID: "
        f"{tokenizer.pad_token_id}"
    )

    test_texts = [
        "{ } [ ] ( ) < > @ # $ % &",
        "What is mitosis?",
        "Write Python code: {x: 10}",
        "Hello, how are you?",
    ]

    for text in test_texts:

        print("\nOriginal:")
        print(text)

        token_ids = tokenizer.encode(text)

        print("Token IDs:")
        print(token_ids)

        decoded = tokenizer.decode(token_ids)

        print("Decoded:")
        print(decoded)

        assert decoded == text, (
            f"Round-trip failed:\n"
            f"Original: {text}\n"
            f"Decoded : {decoded}"
        )

    print("\n" + "=" * 60)
    print("ALL TOKENIZER TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()