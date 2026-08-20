from src.tokenizer.vocabulary import Vocabulary
from src.tokenizer.byte_encoder import ByteEncoder

from src.tokenizer.byte_encoder import ByteEncoder
from src.tokenizer.vocabulary import Vocabulary

encoder = ByteEncoder()

tokens = list(encoder.byte_encoder.values())

vocab = Vocabulary(tokens)

print("Vocabulary size:", len(vocab))

assert len(vocab) == 256


for token in tokens:

    token_id = vocab.get_id(token)

    recovered_token = vocab.get_token(token_id)

    assert recovered_token == token

merges = [
    ("h", "e"),
    ("he", "l"),
    ("hel", "l")
]
for merge in merges:
    left, right = merge
    merged_token = left + right
    vocab.add_token(merged_token)

print("Vocabulary size:", len(vocab))
assert vocab.get_token(256) == "he"
assert vocab.get_token(257) == "hel"
assert vocab.get_token(258) == "hell"
# old_size = len(vocab)

# new_id = vocab.add_token("hello")

# assert new_id == old_size
# assert len(vocab) == old_size + 1

# assert vocab.get_id("hello") == new_id
# assert vocab.get_token(new_id) == "hello"

# same_id = vocab.add_token("hello")

# assert same_id == new_id
# assert len(vocab) == old_size + 1