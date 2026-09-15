import json
from torch.utils.data import Dataset

class PPODataset(Dataset):
    def __init__(self, data_path, max_samples=None):
        self.prompts = []

        with open(data_path, "r", encoding="utf-8",) as f:
            for line in f:
                if not line.strip():
                    continue

                example = json.loads(line)
                prompt = example['prompt'].strip()

                if not prompt:
                    continue

                self.prompts.append(prompt)

                if (max_samples is not None and len(self.prompts) >= max_samples):
                    break

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        return {
            "prompt": self.prompts[index]
        } 

