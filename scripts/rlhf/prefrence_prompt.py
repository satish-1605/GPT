from pathlib import Path
import random
import json

NUM_PROMPTS = 200
INPUT_FILE = Path("data/preference/prompts/preference_prompts.jsonl")
OUTPUT_FILE = Path("data/preference/raw/preference_promtps_200.jsonl")

print("Loading the input file")
with INPUT_FILE.open("r", encoding="utf-8") as f:
    lines = f.readlines()

print("Loading of input file completed")

sampled_lines = random.sample(lines, min(NUM_PROMPTS, len(lines)))

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

print("Loading the output file")
with OUTPUT_FILE.open("w", encoding="utf-8") as f:
    for line in sampled_lines:
        data = json.loads(line)
        data.pop("response", None)
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

print("200 prompts loaded succesfully")