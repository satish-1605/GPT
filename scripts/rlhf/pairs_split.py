import json
import random
from collections import defaultdict

INPUT_FILE = "data/preference/processed/preference_pairs.jsonl"

TRAIN_FILE = "data/preference/processed/train.jsonl"
VAL_FILE = "data/preference/processed/validation.jsonl"
TEST_FILE = "data/preference/processed/test.jsonl"

SEED = 42

pairs = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            pairs.append(json.loads(line))

print(f"Total pairs: {len(pairs)}")


# ---------------------------------------
# 2. Group pairs by prompt_id
# ---------------------------------------

prompt_to_pairs = defaultdict(list)

for pair in pairs:
    prompt_id = pair["prompt_id"]
    prompt_to_pairs[prompt_id].append(pair)

prompt_ids = list(prompt_to_pairs.keys())

print(f"Total unique prompts: {len(prompt_ids)}")


# ---------------------------------------
# 3. Shuffle prompts
# ---------------------------------------

random.seed(SEED)
random.shuffle(prompt_ids)


# ---------------------------------------
# 4. Split PROMPTS, not pairs
# ---------------------------------------

num_prompts = len(prompt_ids)

num_train = int(num_prompts * 0.80)
num_val = int(num_prompts * 0.10)

train_prompts = prompt_ids[:num_train]
val_prompts = prompt_ids[num_train:num_train + num_val]
test_prompts = prompt_ids[num_train + num_val:]


# ---------------------------------------
# 5. Collect all pairs for each split
# ---------------------------------------

train_pairs = []

for prompt_id in train_prompts:
    train_pairs.extend(prompt_to_pairs[prompt_id])

val_pairs = []

for prompt_id in val_prompts:
    val_pairs.extend(prompt_to_pairs[prompt_id])

test_pairs = []

for prompt_id in test_prompts:
    test_pairs.extend(prompt_to_pairs[prompt_id])


# ---------------------------------------
# 6. Write files
# ---------------------------------------

def write_jsonl(filename, data):
    with open(filename, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


write_jsonl(TRAIN_FILE, train_pairs)
write_jsonl(VAL_FILE, val_pairs)
write_jsonl(TEST_FILE, test_pairs)


# ---------------------------------------
# 7. Print statistics
# ---------------------------------------

print("\nDataset split:")
print("-" * 40)

print(
    f"Train:      {len(train_prompts):4d} prompts | "
    f"{len(train_pairs):4d} pairs"
)

print(
    f"Validation: {len(val_prompts):4d} prompts | "
    f"{len(val_pairs):4d} pairs"
)

print(
    f"Test:       {len(test_prompts):4d} prompts | "
    f"{len(test_pairs):4d} pairs"
)

print("-" * 40)

print(f"Output files:")
print(f"  {TRAIN_FILE}")
print(f"  {VAL_FILE}")
print(f"  {TEST_FILE}")

