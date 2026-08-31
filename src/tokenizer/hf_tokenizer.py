from pathlib import Path

from tokenizers import Tokenizer


class HFTokenizer:

    def __init__(self, tokenizer_path: str | Path):

        tokenizer_path = Path(tokenizer_path)

        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"Tokenizer not found: {tokenizer_path}"
            )

        self.tokenizer = Tokenizer.from_file(
            str(tokenizer_path)
        )

        # Use <|endoftext|> as padding for now.
        # We will revisit padding/masking later.
        self.pad_token_id = self.tokenizer.token_to_id(
            "<|endoftext|>"
        )

        if self.pad_token_id is None:
            raise ValueError(
                "<|endoftext|> token not found in tokenizer."
            )

    def encode(self, text: str) -> list[int]:

        return self.tokenizer.encode(text).ids

    def decode(self, token_ids: list[int]) -> str:

        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=False
        )

    def encode_batch(
        self,
        texts: list[str]
    ) -> list[list[int]]:

        return [
            self.encode(text)
            for text in texts
        ]

    def decode_batch(
        self,
        batch_ids: list[list[int]]
    ) -> list[str]:

        return [
            self.decode(ids)
            for ids in batch_ids
        ]