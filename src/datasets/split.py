from sklearn.model_selection import train_test_split


def train_val_test_split(
    documents: list[str],
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """
    Split documents into train, validation, and test sets.

    Args:
        documents: Preprocessed corpus documents.
        train_size: Proportion used for training.
        val_size: Proportion used for validation.
        test_size: Proportion used for testing.
        random_state: Seed for reproducible splitting.

    Returns:
        train_docs, val_docs, test_docs
    """

    if not documents:
        raise ValueError("Documents list cannot be empty.")

    if train_size + val_size + test_size != 1.0:
        raise ValueError(
            "train_size + val_size + test_size must equal 1.0."
        )

    # First split: train vs temporary
    train_docs, temp_docs = train_test_split(
        documents,
        test_size=(val_size + test_size),
        random_state=random_state,
    )

    # Second split: validation vs test
    relative_test_size = test_size / (val_size + test_size)

    val_docs, test_docs = train_test_split(
        temp_docs,
        test_size=relative_test_size,
        random_state=random_state,
    )

    return train_docs, val_docs, test_docs