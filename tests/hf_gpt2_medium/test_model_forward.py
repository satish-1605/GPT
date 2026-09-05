import torch

from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel
)

from src.alignment.sft.instruction_dataset import (
    InstructionDataset
)


# ============================================
# 1. Configuration
# ============================================

MODEL_NAME = "gpt2-medium"

MAX_LENGTH = 1024


# ============================================
# 2. Load tokenizer
# ============================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

tokenizer.pad_token = tokenizer.eos_token


# ============================================
# 3. Load dataset
# ============================================

dataset = InstructionDataset(
    file_path="data/instruction/processed/train.jsonl",
    tokenizer=tokenizer,
    max_length=MAX_LENGTH,
)

print(
    "Total examples:",
    len(dataset)
)


# ============================================
# 4. Load model
# ============================================

model = GPT2LMHeadModel.from_pretrained(
    MODEL_NAME
)

model.config.pad_token_id = (
    tokenizer.pad_token_id
)

print(
    "Parameters:",
    sum(
        p.numel()
        for p in model.parameters()
    )
)


# ============================================
# 5. Get one sample
# ============================================
sample = dataset[0]

input_ids = sample["input_ids"]
labels = sample["labels"]
attention_mask = sample["attention_mask"]

print("Sequence length:", len(input_ids))

# -----------------------------
# Masking statistics
# -----------------------------

prompt_masked = (labels == -100).sum().item()
response_tokens = (labels != -100).sum().item()
padding_tokens = (attention_mask == 0).sum().item()

print("Masked tokens:", prompt_masked)
print("Response tokens:", response_tokens)
print("Padding tokens:", padding_tokens)

# -----------------------------
# Verify response labels
# -----------------------------

response_ids = labels[labels != -100]

decoded_response = tokenizer.decode(
    response_ids.tolist(),
    skip_special_tokens=False
)

print("\nDecoded response labels:")
print(decoded_response)

# -----------------------------
# Verify padding masking
# -----------------------------

padding_labels = labels[attention_mask == 0]

print(
    "\nAll padding labels are -100:",
    torch.all(padding_labels == -100).item()
)

# -----------------------------
# Verify real prompt tokens
# -----------------------------

real_prompt_labels = labels[
    (attention_mask == 1) & (labels == -100)
]

print(
    "Masked real prompt tokens:",
    len(real_prompt_labels)
)

from torch.utils.data import DataLoader

batch_size = 2

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True
)

batch = next(iter(dataloader))

print("Input IDs shape:", batch["input_ids"].shape)
print("Attention mask shape:", batch["attention_mask"].shape)
print("Labels shape:", batch["labels"].shape)


import torch
from torch.optim import AdamW


# ============================================
# 1. CPU
# ============================================

device = torch.device("cpu")

model = model.to(device)
model.train()


# ============================================
# 2. Optimizer
# ============================================

optimizer = AdamW(
    model.parameters(),
    lr=1e-5
)


# ============================================
# 3. Get ONE batch
# ============================================

batch = next(iter(dataloader))

input_ids = batch["input_ids"].to(device)
attention_mask = batch["attention_mask"].to(device)
labels = batch["labels"].to(device)


# ============================================
# 4. Run 10 optimizer steps
# ============================================

losses = []

for step in range(10):

    optimizer.zero_grad()

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )

    loss = outputs.loss

    loss.backward()

    optimizer.step()

    losses.append(loss.item())

    print(
        f"Step {step + 1:02d} | "
        f"Loss: {loss.item():.6f}"
    )


# ============================================
# 5. Check loss trend
# ============================================

print("\nInitial loss:", losses[0])
print("Final loss:", losses[-1])

print(
    "Loss decreased:",
    losses[-1] < losses[0]
)