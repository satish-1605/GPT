from pathlib import Path
import json

class BPETokenizer:

    def __init__(
        self,
        token_to_id: dict[str, int],
        merges: list,
        config: dict,
    ):
        self.token_to_id = token_to_id
        self.id_to_token = {idx:token for token, idx in token_to_id.items()}
        self.merges = merges
        self.config = config

    @staticmethod
    def _load_config(load_dir:Path) -> dict:
        """
        Load tokenizer configuration.
        """
        with (load_dir / "config.json").open("r", encoding="utf-8") as f:
            config = json.load(f)

        return config

    @staticmethod
    def _load_vocabulary(load_dir:Path)-> dict[str, int]:
        """
        Load Token -> ID mapping.
        """
        with (load_dir / "vocab.json").open("r", encoding="utf-8") as f:
                token_to_id = json.load(f)

        return token_to_id
        
    @staticmethod
    def _load_merges(load_dir:Path)-> list[tuple]:
        """
        Load learned merge rules.

        Returns
        -------
        [
            (('l', 'o'), 'lo'),
            (('lo', 'w'), 'low'),
            ...
        ]
        """
        merges = []

        with (load_dir / "merges.txt").open("r", encoding="utf-8") as f:
            for line in f:
                left, right = line.rstrip().split()
                merged_token = left + right

                merges.append(
                    ((left, right), merged_token)
                )
        return merges



    @classmethod
    def from_pretrained(cls, load_dir: str | Path):
        load_dir = Path(load_dir)
        if not load_dir.exists():
            raise FileNotFoundError(f"Tokenizer directory '{load_dir}' does not exist.")  

        config = cls._load_config(load_dir)               

        token_to_id = cls._load_vocabulary(load_dir)

        merges = cls._load_merges(load_dir)

        return cls(token_to_id, merges, config)        

    @staticmethod
    def _tokenize_words(text: str) -> list[str]:
        """
        Split input text into words.
        """
        return text.split()

    @staticmethod
    def _tokenize_characters(word: str) -> list[str]:
        """
        Split a word into character-level tokens.
        """
        return list(word)

    def _apply_merges(self, tokens: list[str])-> list[str]:
        for pair, merged_token in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (i + 1 < len(tokens) 
                    and (tokens[i], tokens[i + 1]) == pair):    
                    new_tokens.append(merged_token)
                    i += 2    
                else:
                    new_tokens.append(tokens[i])
                    i += 1
        
            tokens = new_tokens
        return tokens

    def _convert_to_ids(self, tokens: list[str]) -> list[int]:
        """
        Convert BPE tokens to token IDs.

        Unknown tokens are mapped to the <UNK> token.
        """
        unk_id = self.token_to_id.get("<UNK>")

        if unk_id is None:
            return [self.token_to_id[token] for token in tokens]

        return [self.token_to_id.get(token, unk_id) for token in tokens]


    def encode(self, text: str) -> list[int]:
        """
            Encode raw text into token IDs.

            Pipeline
            --------
            Raw Text
                ↓
            Split into words
                ↓
            Split each word into characters
                ↓
            Apply BPE merges
                ↓
            Convert tokens to IDs
                ↓
            Return flattened list of token IDs
            """

        words = self._tokenize_words(text)

        all_token_ids = []
        for word in words:
            tokens = self._tokenize_characters(word)            
            tokens = self._apply_merges(tokens)
            token_ids = self._convert_to_ids(tokens)
            all_token_ids.extend(token_ids)
        return all_token_ids
        

    def decode(self, token_ids: list[int]) -> str:
        """
        Convert token IDs back into text.

        Parameters
        ----------
        token_ids : list[int]
            Sequence of token IDs.

        Returns
        -------
        str
            Decoded text.
        """
        tokens = []
        for token_id in token_ids:
            if token_id not in self.id_to_token:
                raise ValueError(f"Unknown token ID: {token_id}")
            
            tokens.append(self.id_to_token.get(token_id))

        return " ".join(tokens)


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
