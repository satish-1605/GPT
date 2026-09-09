
import torch
import json
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class PreferenceDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.data = []

        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))

        print(
            f"Loaded {len(self.data)} preference pairs "
            f"from {file_path}"
        )

    def format_prompt(
        self,
        instruction,
        input_text="",
         ):
        if input_text:
            prompt = (
                "### Instruction:\n"
                f"{instruction}\n\n"
                "### Input:\n"
                f"{input_text}\n\n"
                "### Response:\n"
            )
        else:
            prompt = (
                "### Instruction:\n"
                f"{instruction}\n\n"
                "### Response:\n"
            )

        return prompt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        instruction = item['instruction']
        input_text = item['input']

        chosen_response = item["chosen"]["response"]
        rejected_response = item["rejected"]["response"]

        prompt = self.format_prompt(instruction=instruction, input_text=input_text)

        chosen_sequence = (prompt + chosen_response + self.tokenizer.eos_token)
        rejected_sequence = (prompt + rejected_response + self.tokenizer.eos_token)

        chosen_input_ids  = self.tokenizer.encode(
            chosen_sequence, truncation=True, max_length= self.max_length, add_special_tokens=False)

        rejected_input_ids  = self.tokenizer.encode(
            rejected_sequence, truncation=True, max_length= self.max_length, add_special_tokens=False)

        return {
        "chosen_input_ids": chosen_input_ids,
        "rejected_input_ids": rejected_input_ids,
        }
        

def preference_collate_fn(batch, pad_token_id):
    chosen_sequences = [torch.tensor(item['chosen_input_ids'], dtype=torch.long)
                        for item in batch]

    rejected_sequences = [torch.tensor(item['rejected_input_ids'], dtype=torch.long)
                            for item in batch]

    chosen_input_ids = pad_sequence(chosen_sequences, batch_first=True, 
                                    padding_value=pad_token_id)
    
    rejected_input_ids = pad_sequence(rejected_sequences, batch_first=True, 
                                      padding_value=pad_token_id)

    chosen_attention_mask = (chosen_input_ids != pad_token_id).long()
    rejected_attention_mask = (rejected_input_ids != pad_token_id).long()

    return {
        "chosen_input_ids": chosen_input_ids,
        "chosen_attention_mask": chosen_attention_mask,

        "rejected_input_ids": rejected_input_ids,
        "rejected_attention_mask": rejected_attention_mask,
    }