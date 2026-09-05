import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

class InstructionDataset(Dataset):

    def __init__(
        self,
        file_path,
        tokenizer,
        max_length,
    ):

        self.file_path = Path(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenizer.model_max_length = int(1e30)

        # GPT-2 does not have a separate PAD token.
        # Use EOS as PAD and rely on attention_mask
        # to distinguish padding from real tokens.
        self.tokenizer.pad_token = (
            self.tokenizer.eos_token
        )

        self.examples = self._load_dataset()

    # ==========================================================
    # LOAD DATASET
    # ==========================================================

    def _load_dataset(self):

        if not self.file_path.exists():

            raise FileNotFoundError(
                f"Dataset not found: {self.file_path}"
            )

        examples = []

        with self.file_path.open(
            "r",
            encoding="utf-8",
        ) as f:

            for line_number, line in enumerate(
                f,
                start=1,
            ):

                line = line.strip()

                if not line:
                    continue

                try:

                    example = json.loads(line)

                except json.JSONDecodeError as e:

                    raise ValueError(
                        f"Invalid JSON at line "
                        f"{line_number}: {e}"
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

    # ==========================================================
    # FORMAT INSTRUCTION
    # ==========================================================

    def format_instruction(self, example):

        instruction = example[
            "instruction"
        ].strip()

        input_text = example.get(
            "input",
            "",
        ).strip()

        response = example[
            "response"
        ].strip()

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

        return prompt, response

    # ==========================================================
    # ENCODE
    # ==========================================================

    def _encode(
        self,
        prompt,
        response,
    ):
    
        # Tokenize prompt and response without
        # applying model max-length checks.
        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        )["input_ids"]
    
        response_ids = self.tokenizer(
            response,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
        )["input_ids"]
    
        # Explicit EOS after every response
        response_ids.append(
            self.tokenizer.eos_token_id
        )
    
        # ----------------------------------------------------------
        # Reserve space for the response.
        # Keep the beginning of the prompt and response
        # within max_length.
        # ----------------------------------------------------------
    
        available_length = self.max_length
    
        # First combine everything
        token_ids = (
            prompt_ids
            + response_ids
        )
    
        # Truncate final sequence
        token_ids = token_ids[
            :available_length
        ]
    
        # ----------------------------------------------------------
        # Labels
        # ----------------------------------------------------------
    
        prompt_length = len(prompt_ids)
    
        labels = (
            [-100] * prompt_length
            + response_ids
        )
    
        labels = labels[
            :available_length
        ]
    
        # ----------------------------------------------------------
        # Attention mask
        # ----------------------------------------------------------
    
        attention_mask = [
            1
        ] * len(token_ids)
    
        # ----------------------------------------------------------
        # Padding
        # ----------------------------------------------------------
    
        padding_length = (
            self.max_length
            - len(token_ids)
        )
    
        if padding_length > 0:
    
            token_ids += (
                [
                    self.tokenizer.pad_token_id
                ]
                * padding_length
            )
    
            labels += (
                [-100]
                * padding_length
            )
    
            attention_mask += (
                [0]
                * padding_length
            )
    
        return (
            token_ids,
            labels,
            attention_mask,
        )

    # ==========================================================
    # LENGTH
    # ==========================================================

    def __len__(self):

        return len(
            self.examples
        )

    # ==========================================================
    # GET ITEM
    # ==========================================================

    def __getitem__(self, index):

        example = self.examples[
            index
        ]

        prompt, response = (
            self.format_instruction(
                example
            )
        )

        (
            token_ids,
            labels,
            attention_mask,
        ) = self._encode(
            prompt,
            response,
        )

        return {

            "input_ids": torch.tensor(
                token_ids,
                dtype=torch.long,
            ),

            "attention_mask": torch.tensor(
                attention_mask,
                dtype=torch.long,
            ),

            "labels": torch.tensor(
                labels,
                dtype=torch.long,
            ),
        }