
from src.datasets.split import train_val_test_split

def validate_split(
    documents: list[str],
    train_docs: list[str],
    val_docs: list[str],
    test_docs: list[str],
) -> None:
    """
    Validate split sizes and ensure there is no document overlap.
    """

    # Check total number of documents
    assert (
        len(train_docs)
        + len(val_docs)
        + len(test_docs)
        == len(documents)
    ), "Split sizes do not add up to the original dataset."

    # Check for overlap
    train_set = set(train_docs)
    val_set = set(val_docs)
    test_set = set(test_docs)

    assert train_set.isdisjoint(
        val_set
    ), "Train and validation sets overlap."

    assert train_set.isdisjoint(
        test_set
    ), "Train and test sets overlap."

    assert val_set.isdisjoint(
        test_set
    ), "Validation and test sets overlap."


if __name__ == "__main__":

    from src.datasets.preprocess import DatasetPreprocessor
    from src.utils.config import GPTConfig

    config = GPTConfig()

    preprocessor = DatasetPreprocessor()

    documents = preprocessor.preprocess_corpus(
        config.fineweb
    )

    train_docs, val_docs, test_docs = train_val_test_split(
        documents
    )

    validate_split(
        documents,
        train_docs,
        val_docs,
        test_docs,
    )

    print("\nDataset split completed.")
    print(f"Total documents      : {len(documents):,}")
    print(f"Training documents   : {len(train_docs):,}")
    print(f"Validation documents : {len(val_docs):,}")
    print(f"Test documents       : {len(test_docs):,}")