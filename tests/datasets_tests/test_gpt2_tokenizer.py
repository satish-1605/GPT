from src.tokenizer.tokenizer import BPETokenizer

def test_encode_decode_roundtrip():
    tokenizer = BPETokenizer.from_pretrained("artifacts/tokenizer")

    texts = [
        "Hello world!",
        "GPT-2 is a language model.",
        "This is a simple test.",
        "The quick brown fox jumps over the lazy dog.",
    ]

    for text in texts:

        token_ids = tokenizer.encode(text)
        decoded_text = tokenizer.decode(token_ids)

        assert decoded_text == text

def test_unicode_roundtrip():

    tokenizer = BPETokenizer.from_pretrained(
        "artifacts/tokenizer"
    )

    texts = [
        "café",
        "भारत",
        "日本語",
        "Hello भारत 日本語",
        "你好世界",
    ]

    for text in texts:

        token_ids = tokenizer.encode(text)
        decoded_text = tokenizer.decode(token_ids)

        assert decoded_text == text

def test_special_token_roundtrip():

    tokenizer = BPETokenizer.from_pretrained(
        "artifacts/tokenizer"
    )

    text = "Hello<|endoftext|>World"

    token_ids = tokenizer.encode(text)
    decoded_text = tokenizer.decode(token_ids)

    assert decoded_text == text

def test_empty_text():

    tokenizer = BPETokenizer.from_pretrained(
        "artifacts/tokenizer"
    )

    token_ids = tokenizer.encode("")
    decoded_text = tokenizer.decode(token_ids)

    assert token_ids == []
    assert decoded_text == ""

def test_batch_encode_decode():

    tokenizer = BPETokenizer.from_pretrained(
        "artifacts/tokenizer"
    )

    texts = [
        "Hello world!",
        "GPT-2 is powerful.",
        "This is another example."
    ]

    batch_ids = tokenizer.encode_batch(texts)
    decoded_texts = tokenizer.decode_batch(batch_ids)

    assert decoded_texts == texts

def test_encode_decode_roundtrip():

    tokenizer = BPETokenizer.from_pretrained(
        "artifacts/tokenizer"
    )

    texts = [
        "Hello world!",
        "GPT-2 is a language model.",
        "This is a simple test.",
        "The quick brown fox jumps over the lazy dog.",
    ]

    for text in texts:

        token_ids = tokenizer.encode(text)

        assert all(
            0 <= token_id < len(tokenizer.vocabulary)
            for token_id in token_ids
        )

        decoded_text = tokenizer.decode(token_ids)

        assert decoded_text == text


if __name__ == "__main__":
    test_batch_encode_decode()
    test_empty_text()
    test_encode_decode_roundtrip()
    test_special_token_roundtrip()
    test_unicode_roundtrip()
