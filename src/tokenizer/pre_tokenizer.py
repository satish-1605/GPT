import regex

GPT2_PATTERN = regex.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

class GPT2PreTokenizer:

    @staticmethod
    def tokenize(text):
        return GPT2_PATTERN.findall(text)
    
texts = [
    "Hello world!",
    "I'm learning GPT-2.",
    "Don't stop!",
    "Hello,    world!",
    "123 456",
    "भारत में GPT-2",
    "你好 こんにちは",
    "Hello 😊🚀🔥"
]

if __name__ == "__main__":
    pretokenizer = GPT2PreTokenizer()
    for text in texts:
        chunks = GPT2PreTokenizer.tokenize(text)

        print(f"\nText:   {text}")
        print(f"Chunks: {chunks}")