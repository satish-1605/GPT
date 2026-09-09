from pathlib import Path
import time

from src.datasets.preprocess import DatasetPreprocessor
from src.datasets.split import train_val_test_split
from src.tokenizer.gpt2_bpe_trainer import GPT2BPETrainer
from src.utils.config import GPTConfig


config = GPTConfig()


def main():

    print("=" * 60)
    print("GPT-2 BPE Tokenizer Training")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Load and preprocess corpus
    # --------------------------------------------------

    print("\nPreparing training corpus...")

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
    # 3. Initialize GPT-2 BPE trainer
    # --------------------------------------------------

    trainer = GPT2BPETrainer(
        vocab_size=config.model.vocab_size
    )

    # --------------------------------------------------
    # 4. Train tokenizer on TRAIN split only
    # --------------------------------------------------

    print("\nStarting GPT-2 BPE tokenizer training...")
    print(f"Target vocabulary : {config.model.vocab_size}")
    print("-" * 60)

    start = time.time()

    trainer.fit(
        output_dir=config.training.load_dir,
        corpus=train_docs,
        verbose=True,
        log_every=1000
    )

    training_time = time.time() - start

    # --------------------------------------------------
    # 5. Training summary
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("Tokenizer Training Complete")
    print("=" * 60)

    print(
        f"Final vocabulary : "
        f"{len(trainer.vocabulary):,}"
    )

    print(
        f"Number of merges : "
        f"{len(trainer.merges):,}"
    )

    print(
        f"Training time    : "
        f"{training_time / 60:.2f} min"
    )

    print(
        f"Saved to         : "
        f"{Path(config.training.load_dir).resolve()}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()