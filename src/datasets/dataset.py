import torch
from torch.utils.data import Dataset
from typing import List

class GPTDataset(Dataset):
    def __init__(self, 
                 tokenized_stories: List[List[int]], 
                 block_size:int):
        self.tokenized_stories = tokenized_stories
        self.block_size = block_size
        self.samples = []

        self._build_samples()

    def _build_samples(self):
        """
        Convert tokenized stories into GPT training samples.
        """
        for story in self.tokenized_stories:
            if len(story) <= self.block_size:
                continue
            for start in range(len(story)-self.block_size):
                end = start + self.block_size

                input_ids = story[start:end]
                target_ids = story[start+1 : end+1]
                self.samples.append((input_ids, target_ids))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_ids, target_ids = self.samples[idx]

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
        )