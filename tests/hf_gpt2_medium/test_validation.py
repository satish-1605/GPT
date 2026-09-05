
from transformers import AutoTokenizer

from src.alignment.sft.instruction_dataset import InstructionDataset


# ==========================================
# Tokenizer
# ==========================================

tokenizer = AutoTokenizer.from_pretrained(
    "gpt2-medium"
)

tokenizer.pad_token = tokenizer.eos_token


# ==========================================
# Dataset
# ==========================================

dataset = InstructionDataset(
    file_path="data/instruction/processed/train.jsonl",
    tokenizer=tokenizer,
    max_length=1024,
)


# ==========================================
# Dataset Info
# ==========================================

print(
    "\nTotal examples:",
    len(dataset)
)


# ==========================================
# Get First Sample
# ==========================================

sample = dataset[0]

print(sample["input_ids"].shape)
print(sample["attention_mask"].shape)
print(sample["labels"].shape)

print(
    "Attention 1 count:",
    (sample["attention_mask"] == 1).sum().item()
)

print(
    "Attention 0 count:",
    (sample["attention_mask"] == 0).sum().item()
)
