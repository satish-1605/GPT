from src.tokenizer.tokenizer import BPETokenizer
from src.tokenizer.byte_encoder import ByteEncoder
from src.tokenizer.vocabulary import Vocabulary

merges = [
    ("h", "e"),
]

vocabulary_tokens = list(ByteEncoder().byte_encoder.values())

vocab = Vocabulary(vocabulary_tokens)

for merge in merges:
    left, right = merge
    merged_token = left + right
    print(merged_token)
    vocab.add_token(merged_token)

tokenizer = BPETokenizer(merges, vocabulary_tokens=list(vocab.token_to_id.keys()))
texts = [
    "Hello world!",
    "Hello, GPT-2!",
    "café naïve résumé",
    "भारत में GPT-2",
    "你好 こんにちは",
    "Hello 😊🚀🔥",
    "Hello <|endoftext|> mate",
    "<|endoftext|>Hello<|endoftext|>world<|endoftext|>"
]

for text in texts:

    ids = tokenizer.encode(text)

    decoded = tokenizer.decode(ids)

    print(text)
    print(ids)
    print(decoded)
    print()

    assert decoded == text

assert tokenizer.decode(
    tokenizer.encode("hello")
) == "hello"

assert tokenizer.decode(
    tokenizer.encode("भारत")
) == "भारत"

assert tokenizer.decode(
    tokenizer.encode("café")
) == "café"

assert tokenizer.decode(
    tokenizer.encode("Hello 😊🚀")
) == "Hello 😊🚀"


special_id = tokenizer.special_token_to_id["<|endoftext|>"]

print("Special token ID:", special_id)

assert tokenizer.vocabulary.get_token(special_id) == "<|endoftext|>"
assert tokenizer.vocabulary.get_id("<|endoftext|>") == special_id\


special_id = tokenizer.special_token_to_id["<|endoftext|>"]

decoded = tokenizer.decode([special_id])

print(decoded)

assert decoded == "<|endoftext|>"