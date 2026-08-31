import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.alignment.sft.instruction_dataset import InstructionDataset


class MockTokenizer:

    pad_token_id = 0

    def encode(self, text):

        tokens = text.split()

        return list(range(1, len(tokens) + 1))


def create_dataset_file(
    tmp_path: Path,
    examples
):

    file_path = tmp_path / "test.jsonl"

    with file_path.open(
        "w",
        encoding="utf-8"
    ) as f:

        for example in examples:

            f.write(
                json.dumps(example)
                + "\n"
            )

    return file_path


def test_dataloader_batch_shape(tmp_path):

    examples = [
        {
            "instruction": "What is mitosis?",
            "input": "",
            "response": "Cell division."
        },
        {
            "instruction": "What is DNA?",
            "input": "",
            "response": "Genetic material."
        },
        {
            "instruction": "What is RNA?",
            "input": "",
            "response": "Ribonucleic acid."
        },
        {
            "instruction": "What is a cell?",
            "input": "",
            "response": "Basic unit of life."
        }
    ]

    file_path = create_dataset_file(
        tmp_path,
        examples
    )

    max_length = 32
    batch_size = 2

    dataset = InstructionDataset(
        file_path=file_path,
        tokenizer=MockTokenizer(),
        max_length=max_length
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    batch = next(iter(dataloader))

    assert "input_ids" in batch
    assert "labels" in batch

    assert batch["input_ids"].shape == (
        batch_size,
        max_length
    )

    assert batch["labels"].shape == (
        batch_size,
        max_length
    )


def test_dataloader_tensor_dtype(tmp_path):

    examples = [
        {
            "instruction": "What is mitosis?",
            "input": "",
            "response": "Cell division."
        },
        {
            "instruction": "What is DNA?",
            "input": "",
            "response": "Genetic material."
        }
    ]

    file_path = create_dataset_file(
        tmp_path,
        examples
    )

    dataset = InstructionDataset(
        file_path=file_path,
        tokenizer=MockTokenizer(),
        max_length=32
    )

    dataloader = DataLoader(
        dataset,
        batch_size=2
    )

    batch = next(iter(dataloader))

    assert batch["input_ids"].dtype == torch.long
    assert batch["labels"].dtype == torch.long


def test_dataloader_response_masking(
    tmp_path
):

    examples = [
        {
            "instruction": "What is mitosis?",
            "input": "",
            "response": "Cell division."
        },
        {
            "instruction": "What is DNA?",
            "input": "",
            "response": "Genetic material."
        }
    ]

    file_path = create_dataset_file(
        tmp_path,
        examples
    )

    dataset = InstructionDataset(
        file_path=file_path,
        tokenizer=MockTokenizer(),
        max_length=32
    )

    dataloader = DataLoader(
        dataset,
        batch_size=2
    )

    batch = next(iter(dataloader))

    input_ids = batch["input_ids"]
    labels = batch["labels"]

    assert input_ids.shape == labels.shape

    for row, example in enumerate(examples):

        prompt, response = (
            dataset.format_instruction(example)
        )

        prompt_length = len(
            dataset.tokenizer.encode(prompt)
        )

        assert torch.all(
            labels[
                row,
                :prompt_length
            ] == -100
        )


def test_dataloader_padding_masking(
    tmp_path
):

    examples = [
        {
            "instruction": "What is mitosis?",
            "input": "",
            "response": "Cell division."
        },
        {
            "instruction": "Explain.",
            "input": "",
            "response": "A very long response that contains many words."
        }
    ]

    file_path = create_dataset_file(
        tmp_path,
        examples
    )

    dataset = InstructionDataset(
        file_path=file_path,
        tokenizer=MockTokenizer(),
        max_length=32
    )

    dataloader = DataLoader(
        dataset,
        batch_size=2
    )

    batch = next(iter(dataloader))

    input_ids = batch["input_ids"]
    labels = batch["labels"]

    padding_positions = (
        input_ids
        == dataset.tokenizer.pad_token_id
    )

    assert torch.all(
        labels[padding_positions] == -100
    )


def test_dataloader_multiple_batches(
    tmp_path
):

    examples = [
        {
            "instruction": f"Question {i}",
            "input": "",
            "response": f"Answer {i}"
        }
        for i in range(5)
    ]

    file_path = create_dataset_file(
        tmp_path,
        examples
    )

    dataset = InstructionDataset(
        file_path=file_path,
        tokenizer=MockTokenizer(),
        max_length=16
    )

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False
    )

    batches = list(dataloader)

    assert len(batches) == 3

    assert batches[0]["input_ids"].shape == (
        2,
        16
    )

    assert batches[1]["input_ids"].shape == (
        2,
        16
    )

    # Last batch contains one example.
    assert batches[2]["input_ids"].shape == (
        1,
        16
    )