from src.datasets.dataset import GPTDataset
import torch

def test_dataset_length():

    token_stream = list(range(400))

    dataset = GPTDataset(
        token_stream=token_stream,
        context_length=128,
        stride=128
    )

    assert len(dataset) == 3


def test_first_sample():

    token_stream = list(range(400))

    dataset = GPTDataset(
        token_stream=token_stream,
        context_length=128,
        stride=128
    )

    input_ids, target_ids = dataset[0]

    assert input_ids.shape == (128,)
    assert target_ids.shape == (128,)

    assert input_ids.tolist() == list(range(128))
    assert target_ids.tolist() == list(range(1, 129))


def test_middle_sample():

    token_stream = list(range(400))

    dataset = GPTDataset(
        token_stream=token_stream,
        context_length=128,
        stride=128
    )

    input_ids, target_ids = dataset[1]

    assert input_ids.tolist() == list(range(128, 256))
    assert target_ids.tolist() == list(range(129, 257))


def test_last_sample():

    token_stream = list(range(400))

    dataset = GPTDataset(
        token_stream=token_stream,
        context_length=128,
        stride=128
    )

    input_ids, target_ids = dataset[2]

    assert input_ids.tolist() == list(range(256, 384))
    assert target_ids.tolist() == list(range(257, 385))


def test_tensor_dtype():

    token_stream = list(range(400))

    dataset = GPTDataset(
        token_stream=token_stream,
        context_length=128,
        stride=128
    )

    input_ids, target_ids = dataset[0]

    assert input_ids.dtype == torch.long
    assert target_ids.dtype == torch.long

def test_insufficient_tokens():

    token_stream = list(range(128))

    dataset = GPTDataset(
        token_stream=token_stream,
        context_length=128,
        stride=128
    )

    assert len(dataset) == 0

def test_overlapping_windows():

    token_stream = list(range(400))

    dataset = GPTDataset(
        token_stream=token_stream,
        context_length=128,
        stride=64
    )

    assert len(dataset) == 5

    input_ids, target_ids = dataset[1]

    assert input_ids.tolist() == list(range(64, 192))
    assert target_ids.tolist() == list(range(65, 193))

def test_input_target_alignment():

    token_stream = list(range(400))

    dataset = GPTDataset(
        token_stream=token_stream,
        context_length=128,
        stride=128
    )

    for i in range(len(dataset)):

        input_ids, target_ids = dataset[i]

        assert input_ids.shape == target_ids.shape
        assert input_ids.shape == (128,)

        assert torch.equal(
            input_ids[1:],
            target_ids[:-1]
        )

if __name__ == "__main__":
    test_dataset_length()
    test_first_sample()
    test_insufficient_tokens()
    test_tensor_dtype()
    test_middle_sample()
    test_last_sample()
    test_overlapping_windows()
    test_input_target_alignment()
    

