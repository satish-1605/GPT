from collections import Counter

from src.tokenizer.pre_tokenizer import GPT2PreTokenizer
from src.tokenizer.byte_encoder import ByteEncoder

import json
from pathlib import Path

import time

class GPT2BPETrainer:
    def __init__(self, vocab_size:int):
        self.vocab_size = vocab_size

        self.pre_tokenizer = GPT2PreTokenizer()
        self.byte_encoder = ByteEncoder()

        self.corpus = {}

        self.initial_vocabulary = set()
        self.merges = []
        self.merge_ranks ={}

        self.pair_counts = Counter()

        self.special_tokens = ["<|endoftext|>"]

    def initialize_corpus(self, corpus:list[str])-> None:
        """
        Convert raw documents into a GPT-2 byte-level corpus.

        Identical byte-level sequences are stored once with
        their frequency.

        Pipeline:
            raw text
                ↓
            GPT-2 pre-tokenization
                ↓
            byte-level encoding
                ↓
            unique symbol sequences + frequency
        """
        self.corpus = Counter()
        for document in corpus:
            chunks = self.pre_tokenizer.tokenize(document)

            for chunk in chunks:
                byte_text = self.byte_encoder.encode(chunk)
                symbols = tuple(byte_text)
                self.corpus[symbols] += 1

    def initialize_vocabulary(self)->None:
        """
        Build the initial GPT-2 byte-level vocabulary.

        Initial vocabulary:
            - 256 byte-level symbols
            - GPT-2 special tokens
        """
        self.initial_vocabulary = set(self.byte_encoder.byte_encoder.values())

        self.initial_vocabulary.update(
        self.special_tokens
        )
        self.vocabulary = set(self.initial_vocabulary)

    def count_pair_frequencies(self)->None:
        """
        Count frequencies of adjacent symbol pairs
        across the byte-level training corpus.
        """

        self.pair_counts = Counter()

        for sequence, frequency in self.corpus.items():
            for i in range(len(sequence) - 1):
                pair = (sequence[i], sequence[i+1])
                self.pair_counts[pair] += frequency

    def get_best_pair(self):
        """
        Return the most frequent pair.

        If multiple pairs have the same frequency,
        the first encountered pair is selected.
        """

        if not self.pair_counts:
            return None

        best_pair = None
        best_count = -1

        for pair, count in self.pair_counts.items():

            if count > best_count:
                best_pair = pair
                best_count = count

        return best_pair

    def register_merge(self, pair):
        """
        Register a learned BPE merge and assign its rank.
        """

        rank = len(self.merges)

        self.merges.append(pair)
        self.merge_ranks[pair] = rank

        return rank
         

    def merge_pair(self, pair):
        """
        Merge the selected pair throughout the corpus.
        """
        first, second = pair
        merged_symbol = first + second

        new_corpus = Counter()

        for sequnece, frequency in self.corpus.items():
            new_sequence = []
            i=0

            while i < len(sequnece):
                if (
                    i < len(sequnece) -1 
                    and sequnece[i] == first
                    and sequnece[i+1] == second
                ):
                    new_sequence.append(merged_symbol)
                    i+=2
                else:
                    new_sequence.append(sequnece[i])
                    i+=1

            new_corpus[tuple(new_sequence)] += frequency
        self.corpus = new_corpus
        self.vocabulary.add(merged_symbol)

        return merged_symbol

    def train(self, verbose:bool=True, log_every:int=100)->None:
        """
        Train BPE merge rules until the target vocabulary
        size is reached or no mergeable pairs remain.
        """
        start = time.time()
        target_merges = (self.vocab_size - len(self.vocabulary))

        self.count_pair_frequencies()


        if self.vocab_size < len(self.vocabulary):
            raise ValueError(
                f"Target vocabulary size ({self.vocab_size}) "
                f"must be >= initial vocabulary size "
                f"({len(self.vocabulary)})."
            )

        while len(self.vocabulary) < self.vocab_size:
            best_pair = self.get_best_pair()

            if best_pair is None:
                break
            rank = self.register_merge(best_pair)

            previous_vocab_size = len(self.vocabulary)

            merged_symbol = self.merge_pair(best_pair)

            if len(self.vocabulary) == previous_vocab_size:
                break

            completed_merges = len(self.merges)

            self.count_pair_frequencies()

            if verbose and (completed_merges == 1
                 or completed_merges % log_every == 0
                 or len(self.vocabulary) >= self.vocab_size):

                elapsed = time.time() - start

                progress = (
                    completed_merges / target_merges
                ) * 100

                estimated_total = (
                    elapsed / completed_merges
                ) * target_merges

                eta = estimated_total - elapsed

                print(
                    f"Merge: {completed_merges}/{target_merges} "
                    f"({progress:.2f}%) | "
                    f"Vocab: {len(self.vocabulary)} | "
                    f"Unique sequences: {len(self.corpus):,} | "
                    f"Elapsed: {elapsed / 60:.2f} min | "
                    f"ETA: {eta / 60:.2f} min"
                )

    def fit(self, output_dir, corpus:list[str], verbose:bool=True, log_every:int=100)-> None:
        """
        Train the GPT-2 BPE tokenizer on a text corpus.

        Pipeline:
            raw corpus
                ↓
            initialize corpus
                ↓
            initialize vocabulary
                ↓
            BPE training
        """
        self.initialize_corpus(corpus)
        print(
            f"Unique byte sequences: "
            f"{len(self.corpus):,}"
        )
        
        self.initialize_vocabulary()

        self.train(
            verbose=verbose,
            log_every=log_every       
            )
        self.save(output_dir)

    def build_vocab_mapping(self)->dict[str, int]:
        """
        Build a deterministic token → ID mapping.

        Initial vocabulary tokens are assigned IDs first.
        Newly learned BPE tokens are then assigned IDs
        according to merge rank.
        """
        token_to_id = {}
        for token in sorted(self.initial_vocabulary):
            token_to_id[token] = len(token_to_id)

        for first, second in self.merges:
            merged_token = first + second

            if merged_token not in token_to_id:
                token_to_id[merged_token] = len(token_to_id)

        return token_to_id
          

    def _save_vocabulary(self, path: Path) -> None:
        """
        Save token → ID mapping.
        """

        path = Path(path)
        vocabulary = self.build_vocab_mapping()

        with path.open("w", encoding="utf-8") as file:
            json.dump(vocabulary,
                      file,
                      ensure_ascii=False,
                      indent=2)

    def _save_merges(self, path:str|Path)->None:
        """
        Save BPE merge rules in rank order.

        Each line contains:
            first_symbol second_symbol

        The line number determines the merge rank.
        """
        path = Path(path)

        with path.open("w", encoding="utf-8") as file:
            for first, second in self.merges:
                file.write(f"{first} {second}\n")

    def _save_config(self, path:str|Path)-> None:
        """
        Save GPT-2 tokenizer configuration.
        """
        path = Path(path)

        config = {
        "tokenizer_type": "GPT2BPE",
        "vocab_size": len(
            self.build_vocab_mapping()
        ),
        "pre_tokenizer": "GPT2",
        "byte_level": True,
        "special_tokens": self.special_tokens,
        }

        with path.open("w", encoding='utf-8') as file:
            json.dump(config, file, ensure_ascii=False, indent=2)


    def save(self, output_dir:str|Path)-> None:
        """
        Save the trained GPT-2 BPE tokenizer artifacts.

        Saves:
            vocab.json
            merges.txt
            config.json
        """
        output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        self._save_vocabulary(output_dir / "vocab.json")
        self._save_merges(output_dir / "merges.txt")
        self._save_config(output_dir / "config.json")