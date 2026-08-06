from src.tokenizer.tokenizer import BPETokenizer

tokenizer = BPETokenizer.from_pretrained("artifacts/tokenizer")
examples = [
    "lowest",
    "newest",
    "The little cat ran home.",
    "Once upon a time",
]

for text in examples:
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    print(text)
    print(ids)
    print(decoded)

print(len(tokenizer.token_to_id))
print(len(tokenizer.merges))

print(
    tokenizer.encode(
        "Supercalifragilisticexpialidocious"
    )
)
print(list(tokenizer.token_to_id.items())[:30])
print(list(tokenizer.token_to_id.items())[-30:])