from pathlib import Path
import torch

from src.tokenizer.tokenizer import BPETokenizer
from src.datasets.preprocess import DatasetPreprocessor
from src.datasets.split import train_val_test_split
from src.datasets.corpus_tokenizer import CorpusTokenizer
from src.utils.config import GPTConfig


def main():

    config = GPTConfig()

    tokenizer = BPETokenizer.from_pretrained(
        config.training.load_dir
    )

    preprocessor = DatasetPreprocessor()

    documents = preprocessor.preprocess_corpus(
        input_file=config.data.dataset_path
    )

    train_docs, val_docs, test_docs = train_val_test_split(
        documents
    )

    corpus_tokenizer = CorpusTokenizer(tokenizer)

    print("Tokenizing train...")
    train_ids = corpus_tokenizer.tokenize_documents(train_docs)

    print("Tokenizing validation...")
    val_ids = corpus_tokenizer.tokenize_documents(val_docs)

    print("Tokenizing test...")
    test_ids = corpus_tokenizer.tokenize_documents(test_docs)

    output_dir = Path("data/tokenized")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(train_ids, output_dir / "train.pt")
    torch.save(val_ids, output_dir / "val.pt")
    torch.save(test_ids, output_dir / "test.pt")

    print("\nTokenization complete.")
    print(f"Saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()