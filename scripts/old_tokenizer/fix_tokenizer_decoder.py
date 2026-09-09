from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel


TOKENIZER_PATH = Path("artifacts/tokenizer/tokenizer.json")


def main():

    print("=" * 60)
    print("PATCHING TOKENIZER DECODER")
    print("=" * 60)

    tokenizer = Tokenizer.from_file(
        str(TOKENIZER_PATH)
    )

    # Add GPT-2 ByteLevel decoder
    tokenizer.decoder = ByteLevel()

    tokenizer.save(
        str(TOKENIZER_PATH)
    )

    print(f"Tokenizer patched: {TOKENIZER_PATH}")

    # --------------------------------------------------
    # Verify encode -> decode
    # --------------------------------------------------

    text = "Hello world\nThis is a test."

    token_ids = tokenizer.encode(text).ids
    decoded = tokenizer.decode(
        token_ids,
        skip_special_tokens=False
    )

    print("\nOriginal:")
    print(repr(text))

    print("\nDecoded:")
    print(repr(decoded))

    if decoded == text:
        print("\n✅ Tokenizer round-trip PASSED")
    else:
        print("\n❌ Tokenizer round-trip FAILED")


if __name__ == "__main__":
    main()