import torch

from src.datasets.dataset import GPTDataset
from src.datasets.dataloader import create_dataloader


def test_dataloader_shapes():

    token_stream = list(range(2049))

    dataset = GPTDataset(
        token_stream=token_stream,
        context_length=128,
        stride=128
    )

    dataloader = create_dataloader(
        dataset=dataset,
        batch_size=16,
        shuffle=False
    )

    input_ids, target_ids = next(iter(dataloader))
    print(input_ids.shape)
    assert input_ids.shape == (16, 128)
    assert target_ids.shape == (16, 128)

    assert input_ids.dtype == torch.long
    assert target_ids.dtype == torch.long

    

def test_dataloader_alignment():

    token_stream = list(range(1000))

    dataset = GPTDataset(
        token_stream=token_stream,
        context_length=128,
        stride=128
    )

    dataloader = create_dataloader(
        dataset=dataset,
        batch_size=16,
        shuffle=False
    )

    input_ids, target_ids = next(iter(dataloader))

    assert torch.equal(
        input_ids[:, 1:],
        target_ids[:, :-1]
    )

def test_dataloader_batch_shape():

    token_stream = list(range(1000))

    dataset = GPTDataset(
        token_stream=token_stream,
        context_length=128,
        stride=128
    )

    dataloader = create_dataloader(
        dataset=dataset,
        batch_size=16,
        shuffle=False
    )

    input_ids, target_ids = next(iter(dataloader))

    assert input_ids.shape == (16, 128)
    assert target_ids.shape == (16, 128)


def test_dataloader_dtype():

    token_stream = list(range(1000))

    dataset = GPTDataset(
        token_stream=token_stream,
        context_length=128,
        stride=128
    )

    dataloader = create_dataloader(
        dataset=dataset,
        batch_size=16,
        shuffle=False
    )

    input_ids, target_ids = next(iter(dataloader))

    assert input_ids.dtype == torch.long
    assert target_ids.dtype == torch.long


def test_dataloader_alignment():

    token_stream = list(range(1000))

    dataset = GPT2Dataset(
        token_stream=token_stream,
        context_length=128,
        stride=128
    )

    dataloader = create_dataloader(
        dataset=dataset,
        batch_size=16,
        shuffle=False
    )

    input_ids, target_ids = next(iter(dataloader))

    assert torch.equal(
        input_ids[:, 1:],
        target_ids[:, :-1]
    )

if __name__ == "__main__":
    test_dataloader_shapes()
    test_dataloader_alignment()
    test_dataloader_dtype()
    test_dataloader_batch_shape()
