import torch
from torch.utils.data import Dataset
from src.utils.window import build_window_indices
from src.utils.alignment import build_input_target

class GPTDataset(Dataset):
    def __init__(self, 
                 token_stream: list[int], 
                 context_length:int,
                 stride:int):
        self.token_stream = token_stream
        self.context_length = context_length
        self.stride = stride


        self.window_indices = build_window_indices(self.token_stream, 
                                                   self.context_length, 
                                                   self.stride)

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, idx):

        start = self.window_indices[idx]

        input_ids, target_ids = build_input_target(self.token_stream, 
                                                   start=start,
                                                   context_length=self.context_length)

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
        )