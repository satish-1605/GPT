from pathlib import Path
import json

from src.tokenizer.pre_tokenizer import GPT2PreTokenizer
from src.tokenizer.byte_encoder import ByteEncoder
from src.tokenizer.bpe import BPE
from src.tokenizer.vocabulary import Vocabulary

class BPETokenizer:

    def __init__(
        self,
        merges,
        vocabulary_tokens
    ):
        self.pre_tokenizer = GPT2PreTokenizer()
        self.byte_encoder = ByteEncoder()
        self.bpe = BPE(merges)
        self.vocabulary = Vocabulary(vocabulary_tokens)

        self.special_tokens = ["<|endoftext|>"]

        self.special_token_to_id = {}

        for special_token in self.special_tokens:
            token_id = self.vocabulary.add_token(special_token)

            self.special_token_to_id[special_token] = token_id

    @classmethod
    def from_pretrained(cls, load_dir: str | Path):
        """
        Load a trained GPT-2 BPE tokenizer from disk.

        Expected files:

            vocab.json
            merges.txt
            config.json
        """
        load_dir = Path(load_dir)
        if not load_dir.exists():
            raise FileNotFoundError(f"Tokenizer directory '{load_dir}' does not exist.")  

        vocab_path = load_dir / "vocab.json"
        merges_path = load_dir / "merges.txt"
        config_path = load_dir / "config.json"

        if not vocab_path.exists():
            raise FileNotFoundError(
                f"Vocabulary file not found: {vocab_path}"
            )

        if not merges_path.exists():
            raise FileNotFoundError(
                f"Merges file not found: {merges_path}"
            )

        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {config_path}"
            )

        with vocab_path.open("r", encoding="utf-8") as file:
            vocabulary = json.load(file)              

        merges = []
        with merges_path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ", 1)
                if len(parts) != 2:
                    raise ValueError(
                        f"Invalid merge rule: {line}"
                    )

                first, second = parts
                merges.append(
                    (first, second)
                )

        with config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)

        if config.get("tokenizer_type") != "GPT2BPE":
            raise ValueError(
                f"Unsupported tokenizer type: "
                f"{config.get('tokenizer_type')}"
            )

        if config.get("vocab_size") != len(vocabulary):
            raise ValueError(
                "Vocabulary size in config.json does not "
                "match vocab.json."
            )

        vocabulary_tokens = [token for token, token_id in sorted(vocabulary.items(), 
                                                                 key=lambda item: item[1])]

        tokenizer = cls(
        merges=merges,
        vocabulary_tokens=vocabulary_tokens
                )

        return tokenizer 


    def encode(self, text: str) -> list[int]:

        token_ids = []
        parts = text.split("<|endoftext|>")

        for i, part in enumerate(parts):
            if part:
                chunks = self.pre_tokenizer.tokenize(part)
                for chunk in chunks:

                    byte_text = self.byte_encoder.encode(chunk)

                    symbols = list(byte_text)

                    bpe_tokens = self.bpe.apply_bpe(symbols)         

                    for token in bpe_tokens:
                        token_id = self.vocabulary.get_id(token)
                        token_ids.append(token_id)

            if i < len(parts) - 1:

                special_id = self.special_token_to_id[
                    "<|endoftext|>"
                ]

                token_ids.append(special_id)

        return token_ids
        

    def decode(self, token_ids: list[int]) -> str:

        decoded_parts = []
        normal_tokens = []

        special_token_ids = set(self.special_token_to_id.values())

        for token_id in token_ids:

            token = self.vocabulary.get_token(token_id)

            if token_id in special_token_ids:

                if normal_tokens:
                    byte_text = "".join(normal_tokens)
                    decoded_parts.append(
                        self.byte_encoder.decode(byte_text)
                    )
                    normal_tokens = []

                decoded_parts.append(token)

            else:
                normal_tokens.append(token)

        if normal_tokens:
            byte_text = "".join(normal_tokens)
            decoded_parts.append(
                self.byte_encoder.decode(byte_text)
            )

        return "".join(decoded_parts)


    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        """
        Encode multiple texts.
        """

        return [
            self.encode(text)
            for text in texts
        ]

    def decode_batch(self, batch_ids: list[list[int]]) -> list[str]:
        """
        Decode multiple token ID sequences.
        """

        return [
            self.decode(ids)
            for ids in batch_ids
        ]

