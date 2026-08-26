from src.datasets.dataloader import create_dataloader
from src.datasets.dataset import GPTDataset

from src.tokenizer.tokenizer import BPETokenizer
from src.datasets.preprocess import DatasetPreprocessor
from src.datasets.split import train_val_test_split
from src.utils.config import GPTConfig
from src.datasets.corpus_tokenizer import CorpusTokenizer
import torch


def get_train_val_test_loaders(config: GPTConfig):

    # -------------------------
    # Load cached token IDs
    # -------------------------

    train_ids = torch.load(
        "data/tokenized/train.pt",
        weights_only=False
    )

    val_ids = torch.load(
        "data/tokenized/val.pt",
        weights_only=False
    )

    test_ids = torch.load(
        "data/tokenized/test.pt",
        weights_only=False
    )

    # -------------------------
    # Token streams
    # -------------------------

    tokenizer = BPETokenizer.from_pretrained(
        config.training.load_dir
    )

    corpus_tokenizer = CorpusTokenizer(tokenizer)

    train_stream = corpus_tokenizer.build_token_stream(train_ids)
    val_stream = corpus_tokenizer.build_token_stream(val_ids)
    test_stream = corpus_tokenizer.build_token_stream(test_ids)

    # -------------------------
    # Datasets
    # -------------------------

    train_dataset = GPTDataset(
        token_stream=train_stream,
        context_length=config.data.context_length,
        stride=config.data.stride,
    )

    val_dataset = GPTDataset(
        token_stream=val_stream,
        context_length=config.data.context_length,
        stride=config.data.stride,
    )

    test_dataset = GPTDataset(
        token_stream=test_stream,
        context_length=config.data.context_length,
        stride=config.data.stride,
    )

    # -------------------------
    # DataLoaders
    # -------------------------

    train_loader = create_dataloader(
        dataset=train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
    )

    val_loader = create_dataloader(
        dataset=val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
    )

    test_loader = create_dataloader(
        dataset=test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader