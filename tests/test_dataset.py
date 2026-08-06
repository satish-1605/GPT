from src.datasets.dataset import GPTDataset

tokenized_stories = [
    [2, 10, 20, 30, 40, 50, 3],
    [2, 60, 70, 80, 3]
]

ds = GPTDataset(tokenized_stories, block_size=4)
# ds._build_samples()
# print(ds.samples)
# print(len(ds))
# print(ds[0])

from src.datasets.dataloader import DataLoader

dl = DataLoader(ds, 2)
for input_ids, target_ids in dl:
    print(input_ids, target_ids)
    break




