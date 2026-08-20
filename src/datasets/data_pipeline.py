from src.datasets.dataloader import create_dataloader
from src.datasets.dataset import GPTDataset

from src.tokenizer.tokenizer import BPETokenizer
from src.datasets.preprocess import DatasetPreprocessor
from src.datasets.split import train_val_test_split
from src.utils.config import GPTConfig
from src.datasets.tokenize import CorpusTokenizer


def get_train_val_test_loaders(config: GPTConfig):

    tokenizer = BPETokenizer.from_pretrained(
        config.load_dir
    )

    preprocessor = DatasetPreprocessor()

    documents = preprocessor.preprocess_corpus(
        input_file=config.fineweb_dataset_path
    )

    train_docs, val_docs, test_docs = train_val_test_split(
        documents
    )

    corpus_tokenizer = CorpusTokenizer(tokenizer)

    train_ids = corpus_tokenizer.tokenize_documents(train_docs)
    val_ids = corpus_tokenizer.tokenize_documents(val_docs)
    test_ids = corpus_tokenizer.tokenize_documents(test_docs)

    train_stream = corpus_tokenizer.build_token_stream(train_ids)
    val_stream = corpus_tokenizer.build_token_stream(val_ids)
    test_stream = corpus_tokenizer.build_token_stream(test_ids)



    # -------------------------
    # Datasets
    # -------------------------

    train_dataset = GPTDataset(
        token_stream=train_stream,
        context_length=config.context_length,
        stride=config.stride,
    )

    val_dataset = GPTDataset(
        token_stream=val_stream,
        context_length=config.context_length,
        stride=config.stride,
    )

    test_dataset = GPTDataset(
        token_stream=test_stream,
        context_length=config.context_length,
        stride=config.stride,
    )

    # -------------------------
    # DataLoaders
    # -------------------------

    train_loader = create_dataloader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )

    val_loader = create_dataloader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    test_loader = create_dataloader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )