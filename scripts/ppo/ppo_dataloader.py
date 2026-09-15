from torch.utils.data import DataLoader
from scripts.ppo.ppo_dataset import PPODataset

def create_ppo_dataloader(data_path, batch_size=4, max_samples=None, shuffle=True):
    dataset = PPODataset(data_path=data_path,
                         max_samples=max_samples,
                         )

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle)

    return dataloader