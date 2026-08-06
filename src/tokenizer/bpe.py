from collections import Counter
from typing import List, Optional
from pathlib import Path
import json
import time

SPECIAL_TOKENS = [
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>",
]


class BPETrainer:
    """
    Byte Pair Encoding (BPE) Trainer.

    Responsibilities
    ----------------
    1. Build the initial character-level corpus
    2. Initialize the vocabulary
    3. Count adjacent symbol pair frequencies
    4. Find the best pair
    5. Merge the best pair
    6. Repeat until the vocabulary reaches the desired size
    """

    def __init__(self, vocab_size: int):

        self.vocab_size = vocab_size

        # Character-level corpus
        self.corpus: List[List[str]] = []

        # Learned vocabulary
        self.vocabulary: set[str] = set()

        # Ordered merge rules
        self.merges: List[tuple] = []

        # Pair frequencies
        self.pair_counts: Counter = Counter()

        self.special_tokens = [
                "<PAD>",
                "<UNK>",
                "<BOS>",
                "<EOS>",
            ]

    def initialize_corpus(self, corpus: List[str]) -> None:
        """
        Convert raw text into a character-level corpus.

        Example
        -------
        Input:
            ["low lower"]

        Output:
            [
                ['l', 'o', 'w'],
                ['l', 'o', 'w', 'e', 'r']
            ]
        """

        tokenized_words = []

        for story in corpus:

            words = story.split()

            for word in words:
                tokenized_words.append(list(word))

        self.corpus = tokenized_words

    def initialize_vocabulary(self) -> None:
        """
        Build the initial vocabulary consisting of
        all unique characters.
        """

        self.vocabulary = set(self.special_tokens)

        for word in self.corpus:

            for symbol in word:

                self.vocabulary.add(symbol)

    def count_pair_frequencies(self) -> None:
        """
        Count adjacent symbol pair frequencies.
        """

        self.pair_counts = Counter()

        for word in self.corpus:

            for i in range(len(word) - 1):

                pair = (word[i], word[i + 1])

                self.pair_counts[pair] += 1

    def get_best_pair(self) -> Optional[tuple]:
        """
        Return the most frequent pair.
        """

        if not self.pair_counts:
            return None

        return self.pair_counts.most_common(1)[0][0]

    def merge_pair(self, pair: tuple) -> None:
        """
        Merge a symbol pair throughout the corpus.

        Example
        -------
        Pair:
            ('l', 'o')

        Before:
            ['l', 'o', 'w']

        After:
            ['lo', 'w']
        """

        merged_corpus = []

        for word in self.corpus:

            new_word = []

            i = 0

            while i < len(word):

                if (
                    i + 1 < len(word)
                    and (word[i], word[i + 1]) == pair
                ):

                    new_word.append(word[i] + word[i + 1])
                    i += 2

                else:

                    new_word.append(word[i])
                    i += 1

            merged_corpus.append(new_word)

        self.corpus = merged_corpus
        merged_token = "".join(pair)
        self.merges.append((pair, merged_token))

    def train_step(self) -> Optional[tuple]:
        """
        Perform one BPE merge iteration.

        Returns
        -------
        tuple
            Best pair selected.

        None
            If no more merges are possible.
        """

        self.count_pair_frequencies()

        best_pair = self.get_best_pair()

        if best_pair is None:
            return None

        self.merge_pair(best_pair)

        return best_pair

    def fit(self, corpus: List[str], verbose:bool=False, log_every:int=100) -> None:
        """
        Train the BPE tokenizer.
        """

        self.initialize_corpus(corpus)

        self.initialize_vocabulary()
        start_time = time.time()
        initial_vocab_size = len(self.vocabulary)

        if verbose:
            print("=" * 60)
            print("Training BPE Tokenizer")
            print("=" * 60)
            print(f"Stories Loaded     : {len(corpus):,}")
            print(f"Target Vocabulary : {self.vocab_size:,}")
            print(f"Initial Vocabulary: {initial_vocab_size:,}")
            print("=" * 60)

        if self.vocab_size <= len(self.vocabulary):
            raise ValueError(
                f"Vocabulary size ({self.vocab_size}) must be greater than "
                f"the initial vocabulary ({len(self.vocabulary)})."
            )

        while len(self.vocabulary) < self.vocab_size:

            pair = self.train_step()

            if pair is None:
                break

            merged_token = "".join(pair)

            self.vocabulary.add(merged_token)

            if verbose and len(self.vocabulary) % log_every == 0:
                elapsed = time.time() - start_time
                progress = (
                    len(self.vocabulary) / self.vocab_size
                ) * 100
                eta = (
                    elapsed / len(self.vocabulary)
                ) * (
                    self.vocab_size - len(self.vocabulary)
                )
                print(
                    f"Vocabulary: {len(self.vocabulary):5d}/{self.vocab_size} | "
                    f"Progress: {progress:6.2f}% | "
                    f"Merges: {len(self.merges):5d} | "
                    f"Elapsed: {elapsed/60:.2f} min | "
                    f"ETA: {eta/60:.2f} min"
                )
        if verbose:
            total_time = time.time() - start_time

            print("=" * 60)
            print("Training Complete")
            print("=" * 60)
            print(f"Final Vocabulary : {len(self.vocabulary)}")
            print(f"Total Merges     : {len(self.merges)}")
            print(f"Total Time       : {total_time/60:.2f} minutes")

    # -------------------------
    # Serialization
    # -------------------------
    def _create_token_to_id_mapping(self) -> dict[str, int]:
        """
        Build a deterministic Token -> ID mapping.

        The vocabulary is sorted before assigning IDs so that
        the mapping remains reproducible across runs.
        """

        return {
            token: idx
            for idx, token in enumerate(sorted(self.vocabulary))
        }

    def _save_vocabulary(self, save_dir:Path, token_to_id:dict[str, int]) -> Path:
        vocab_file = save_dir / "vocab.json"
    
        with vocab_file.open("w", encoding="utf-8") as f:
            json.dump(token_to_id, f, indent=4, ensure_ascii=False)

    def _save_merges(self, save_dir:Path) -> Path:
        """
        Save learned merge rules.

        Each line contains one merge pair.

        Example
        -------
        l o
        lo w
        low e
        """
        merge_file = save_dir / "merges.txt"

        with merge_file.open("w", encoding="utf-8") as f:
            for (left, right), merged_token in self.merges:
                f.write(f"{left} {right}\n")


    def _save_config(self, save_dir:Path)->Path:
        """
        Save tokenizer configuration.
        """
        config_file = save_dir / "config.json"

        config = {
        "algorithm": "BPE",
        "vocab_size": self.vocab_size,
        "version": "1.0",
        "special_tokens": self.special_tokens,
        }

        with config_file.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)


    def save(self, save_path: str | Path) -> None:
        """
        Save the trained tokenizer artifacts.

        Directory Structure
        -------------------
        save_path/
            vocab.json
            merges.txt
            config.json
        """

        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        token_to_id = self._create_token_to_id_mapping()

        self._save_vocabulary(save_dir, token_to_id)

        self._save_merges(save_dir) 

        self._save_config(save_dir)  

    # -------------------------
    # Deserialization
    # -------------------------
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
    def load(cls, load_dir:str | Path) -> "BPETrainer":
        """
        Load a BPE trainer from disk.   
        """
        load_dir = Path(load_dir)
        if not load_dir.exists():
            raise FileNotFoundError(f"Tokenizer directory '{load_dir}' does not exist.")      
        

        config = cls._load_config(load_dir)

        trainer = cls(vocab_size=config["vocab_size"])
        trainer.special_tokens = config["special_tokens"]

        token_to_id = cls._load_vocabulary(load_dir)

        trainer.vocabulary = set(token_to_id.keys())

        trainer.merges = cls._load_merges(load_dir)

        return trainer 

corpus = ["low lowest lower"]

if __name__ =="__main__":
    trainer = BPETrainer(25)
    trainer.fit(corpus)
    trainer.save("artifacts/tokenizer")
