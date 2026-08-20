from src.tokenizer.byte_encoder import ByteEncoder

encoder = ByteEncoder()

texts = ["Hello world!", "Hello, GPT-2!", "café naïve résumé", "भारत में GPT-2", 
         "你好 こんにちは", "Hello 😊🚀🔥"]


for text in texts:
    encoded = encoder.encode(text)
    decoded = encoder.decode(encoded)

    print(f"Original : {text}")
    print(f"Encoded  : {encoded}")
    print(f"Decoded  : {decoded}")
    print("-" * 50)

    assert decoded == text

assert len(encoder.byte_encoder) == 256
assert len(encoder.byte_decoder) == 256

for byte in range(256):
    encoded = encoder.byte_encoder[byte]
    decoded = encoder.byte_decoder[encoded]

    assert decoded == byte