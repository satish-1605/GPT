import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

class InstructionDataset(Dataset):
    def __init__(self, file_path:str | Path, tokenizer, max_length:int):
        self.file_path = Path(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.examples = self._load_dataset()

    def _load_dataset(self):

        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.file_path}"
            )
        
        examples = []

        with self.file_path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    example = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid JSON at line {line_number}:{e}"
                    )

                if "instruction" not in example:
                    raise ValueError(
                        f"Missing 'instruction' "
                        f"at line {line_number}"
                    )

                if "response" not in example:
                    raise ValueError(
                        f"Missing 'response' "
                        f"at line {line_number}"
                    )

                examples.append(example)
        return examples

    def format_instruction(self, example):
        instruction = example["instruction"].strip()
        input_text = example.get("input", "").strip()
        response = example["response"].strip()

        if input_text:
            prompt =  (
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

        response_text = (
            f"{response}"
        "<|endoftext|>")

        return prompt, response_text

    def _encode(self, prompt, response):
        prompt_ids = self.tokenizer.encode(prompt)
        response_ids = self.tokenizer.encode(response)

        token_ids = prompt_ids + response_ids
        token_ids = token_ids[:self.max_length]

        prompt_length = len(prompt_ids)

        labels = (
        [-100] * prompt_length
        + response_ids
    )
        labels = labels[:self.max_length]

        padding_length = (self.max_length - len(token_ids))

        if padding_length > 0:
            pad_id = self.tokenizer.pad_token_id

            token_ids = token_ids + (
                [pad_id]
                * padding_length
            )

            labels = labels + (
                [-100] * padding_length
            )

        return token_ids, labels

    def __len__(self):

        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        prompt, response  = self.format_instruction(example)

        token_ids, labels = self._encode(prompt, response)

        input_ids = torch.tensor(token_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": labels,
        }