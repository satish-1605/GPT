import json
from pathlib import Path

import torch

from src.alignment.sft.instruction_dataset import InstructionDataset


class MockTokenizer:

    pad_token_id = 0

    def encode(self, text):

        # Simple deterministic tokenizer for testing.
        # Each whitespace-separated token gets an ID.
        tokens = text.split()

        return list(range(1, len(tokens) + 1))


def create_dataset_file(tmp_path: Path, examples):

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


def test_dataset_length(tmp_path):

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

    assert len(dataset) == 2


def test_input_is_included(tmp_path):

    examples = [
        {
            "instruction": "Summarize this.",
            "input": "The sky is blue.",
            "response": "The sky has a blue color."
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

    prompt, response = dataset.format_instruction(
        examples[0]
    )

    assert "### Instruction:" in prompt
    assert "Summarize this." in prompt
    assert "### Input:" in prompt
    assert "The sky is blue." in prompt
    assert "### Response:" in prompt

    assert response.startswith(
        "The sky has a blue color."
    )

    assert response.endswith(
        "<|endoftext|>"
    )


def test_empty_input_is_handled(tmp_path):

    examples = [
        {
            "instruction": "What is mitosis?",
            "input": "",
            "response": "Cell division."
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

    prompt, response = dataset.format_instruction(
        examples[0]
    )

    assert "### Input:" not in prompt
    assert "What is mitosis?" in prompt
    assert "### Response:" in prompt


def test_response_only_loss_masking(tmp_path):

    examples = [
        {
            "instruction": "What is mitosis?",
            "input": "",
            "response": "Cell division."
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

    item = dataset[0]

    input_ids = item["input_ids"]
    labels = item["labels"]

    assert isinstance(
        input_ids,
        torch.Tensor
    )

    assert isinstance(
        labels,
        torch.Tensor
    )

    assert input_ids.shape == labels.shape

    # Prompt tokens must be masked.
    prompt, response = dataset.format_instruction(
        examples[0]
    )

    prompt_length = len(
        dataset.tokenizer.encode(prompt)
    )

    assert torch.all(
        labels[:prompt_length] == -100
    )


def test_response_tokens_are_not_masked(tmp_path):

    examples = [
        {
            "instruction": "What is mitosis?",
            "input": "",
            "response": "Cell division."
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

    item = dataset[0]

    labels = item["labels"]

    prompt, response = dataset.format_instruction(
        examples[0]
    )

    prompt_length = len(
        dataset.tokenizer.encode(prompt)
    )

    response_ids = dataset.tokenizer.encode(
        response
    )

    assert torch.equal(
        labels[
            prompt_length:
            prompt_length + len(response_ids)
        ],
        torch.tensor(
            response_ids,
            dtype=torch.long
        )
    )


def test_padding_is_masked(tmp_path):

    examples = [
        {
            "instruction": "What is mitosis?",
            "input": "",
            "response": "Cell division."
        }
    ]

    file_path = create_dataset_file(
        tmp_path,
        examples
    )

    max_length = 32

    dataset = InstructionDataset(
        file_path=file_path,
        tokenizer=MockTokenizer(),
        max_length=max_length
    )

    item = dataset[0]

    input_ids = item["input_ids"]
    labels = item["labels"]

    assert len(input_ids) == max_length
    assert len(labels) == max_length

    padding_positions = (
        input_ids == dataset.tokenizer.pad_token_id
    )

    assert torch.all(
        labels[padding_positions] == -100
    )


def test_max_length_truncation(tmp_path):

    examples = [
        {
            "instruction": "What is mitosis?",
            "input": "",
            "response": "Cell division happens in cells."
        }
    ]

    file_path = create_dataset_file(
        tmp_path,
        examples
    )

    max_length = 8

    dataset = InstructionDataset(
        file_path=file_path,
        tokenizer=MockTokenizer(),
        max_length=max_length
    )

    item = dataset[0]

    assert len(item["input_ids"]) == max_length
    assert len(item["labels"]) == max_length


def test_missing_instruction_raises_error(tmp_path):

    examples = [
        {
            "input": "",
            "response": "Cell division."
        }
    ]

    file_path = create_dataset_file(
        tmp_path,
        examples
    )

    try:

        InstructionDataset(
            file_path=file_path,
            tokenizer=MockTokenizer(),
            max_length=32
        )

        assert False, (
            "Expected ValueError"
        )

    except ValueError as e:

        assert "instruction" in str(e)


def test_missing_response_raises_error(tmp_path):

    examples = [
        {
            "instruction": "What is mitosis?",
            "input": ""
        }
    ]

    file_path = create_dataset_file(
        tmp_path,
        examples
    )

    try:

        InstructionDataset(
            file_path=file_path,
            tokenizer=MockTokenizer(),
            max_length=32
        )

        assert False, (
            "Expected ValueError"
        )

    except ValueError as e:

        assert "response" in str(e)