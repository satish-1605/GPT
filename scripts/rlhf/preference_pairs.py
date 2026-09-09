import json
from itertools import combinations

CANDIDATES_FILE = "data/preference/raw/candidates.jsonl"
RANKING_FILE = "data/preference/rankings/human_ranking.jsonl"
OUTPUT_FILE = "data/preference/processed/preference_pairs.jsonl"

candidates_data = {}

with open(CANDIDATES_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        item = json.loads(line)
        candidates_data[item["id"]] = item


# -----------------------------
# Load rankings and create pairs
# -----------------------------
total_pairs = 0

with open(RANKING_FILE, "r", encoding="utf-8") as f_in, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:

    for line in f_in:
        if not line.strip():
            continue

        ranking_data = json.loads(line)

        prompt_id = ranking_data["id"]
        ranking = ranking_data["ranking"]

        # Find corresponding candidate example
        if prompt_id not in candidates_data:
            print(f"Warning: No candidates found for id={prompt_id}")
            continue

        candidate_data = candidates_data[prompt_id]

        instruction = candidate_data.get("instruction", "")
        input_text = candidate_data.get("input", "")

        # Map candidate_id -> candidate object
        candidates = {
            candidate["candidate_id"]: candidate
            for candidate in candidate_data["candidates"]
        }

        # Generate all pairs according to ranking
        #
        # Example:
        # ["A", "B", "D", "C"]
        #
        # => A>B, A>D, A>C, B>D, B>C, D>C
        for chosen_id, rejected_id in combinations(ranking, 2):

            # Validate candidate IDs
            if chosen_id not in candidates:
                print(
                    f"Warning: candidate {chosen_id} "
                    f"not found for id={prompt_id}"
                )
                continue

            if rejected_id not in candidates:
                print(
                    f"Warning: candidate {rejected_id} "
                    f"not found for id={prompt_id}"
                )
                continue

            pair = {
                "prompt_id": prompt_id,
                "instruction": instruction,
                "input": input_text,
                "chosen": {
                    "candidate_id": chosen_id,
                    "response": candidates[chosen_id]["response"]
                },
                "rejected": {
                    "candidate_id": rejected_id,
                    "response": candidates[rejected_id]["response"]
                }
            }

            f_out.write(
                json.dumps(pair, ensure_ascii=False) + "\n"
            )

            total_pairs += 1


print(f"Created {OUTPUT_FILE}")
print(f"Total pairs: {total_pairs}")