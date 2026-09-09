from pathlib import Path
import json
import time

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from src.datasets.preprocess import DatasetPreprocessor
from src.datasets.split import train_val_test_split
from src.utils.config import GPTConfig
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


config = GPTConfig()


def main():

    print("=" * 60)
    print("GPT-2 BPE Tokenizer Training — Fast")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Load corpus
    # --------------------------------------------------

    print("\nPreparing corpus...")

    start = time.time()

    preprocessor = DatasetPreprocessor()

    documents = preprocessor.preprocess_corpus(
        input_file=config.data.dataset_path
    )

    print(f"Documents loaded : {len(documents):,}")

    # --------------------------------------------------
    # 2. Train / Validation / Test split
    # --------------------------------------------------

    train_docs, val_docs, test_docs = train_val_test_split(
        documents
    )

    print(f"Train documents  : {len(train_docs):,}")
    print(f"Val documents    : {len(val_docs):,}")
    print(f"Test documents   : {len(test_docs):,}")

    print(
        f"Preparation time : "
        f"{time.time() - start:.2f} sec"
    )

    # --------------------------------------------------
    # 3. Save training corpus temporarily
    # --------------------------------------------------

    temp_file = Path(
        "data/processed/tokenizer_train.txt"
    )

    temp_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nWriting tokenizer training corpus...")

    with temp_file.open(
        "w",
        encoding="utf-8"
    ) as f:

        for document in train_docs:

            f.write(document)

            f.write("\n")

    print(
        f"Training corpus saved to: "
        f"{temp_file}"
    )

    # --------------------------------------------------
    # 4. Create tokenizer
    # --------------------------------------------------

    tokenizer = Tokenizer(
        BPE(
            unk_token="<|endoftext|>"
        )
    )

    tokenizer.pre_tokenizer = ByteLevel(
        add_prefix_space=False
    )

    tokenizer.decoder = ByteLevelDecoder()

    # --------------------------------------------------
    # 5. Trainer
    # --------------------------------------------------

    trainer = BpeTrainer(
        vocab_size=config.model.vocab_size,
        min_frequency=2,
        special_tokens=[
            "<|endoftext|>"
        ],
        show_progress=True
    )

    # --------------------------------------------------
    # 6. Train
    # --------------------------------------------------

    print("\nStarting tokenizer training...")

    start = time.time()

    tokenizer.train(
        files=[
            str(temp_file)
        ],
        trainer=trainer
    )

    training_time = time.time() - start

    # --------------------------------------------------
    # 7. Save tokenizer
    # --------------------------------------------------

    output_dir = Path(
        config.training.load_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    tokenizer_json = (
        output_dir / "tokenizer.json"
    )

    tokenizer.save(
        str(tokenizer_json)
    )

    # --------------------------------------------------
    # 8. Save vocab.json
    # --------------------------------------------------

    vocab = tokenizer.get_vocab()

    vocab = dict(
        sorted(
            vocab.items(),
            key=lambda x: x[1]
        )
    )

    with (
        output_dir / "vocab.json"
    ).open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            vocab,
            f,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------
    # 9. Save config
    # --------------------------------------------------

    tokenizer_config = {
        "tokenizer_type": "GPT2BPE",
        "vocab_size": len(vocab),
        "pre_tokenizer": "GPT2",
        "byte_level": True,
        "special_tokens": [
            "<|endoftext|>"
        ]
    }

    with (
        output_dir / "config.json"
    ).open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            tokenizer_config,
            f,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------
    # 10. Summary
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("Tokenizer Training Complete")
    print("=" * 60)

    print(
        f"Vocabulary size : {len(vocab):,}"
    )

    print(
        f"Training time   : "
        f"{training_time / 60:.2f} min"
    )

    print(
        f"Saved to        : "
        f"{output_dir.resolve()}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()