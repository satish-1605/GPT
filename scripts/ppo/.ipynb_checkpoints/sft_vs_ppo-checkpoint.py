import json
from pathlib import Path


SFT_PATH = "artifacts/ppo/sft_baseline.jsonl"
PPO_PATH = "artifacts/ppo/ppo_evaluation.jsonl"

OUTPUT_PATH = "artifacts/ppo/sft_vs_ppo_comparison.jsonl"

IMPROVEMENT_THRESHOLD = 0.05


def load_results(path):
    results = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            example = json.loads(line)

            index = example["index"]

            results[index] = example

    return results


def main():
    print("=" * 70)
    print("SFT vs PPO COMPARISON")
    print("=" * 70)
    print()

    print(f"SFT results : {SFT_PATH}")
    print(f"PPO results : {PPO_PATH}")
    print()

    sft_results = load_results(SFT_PATH)
    ppo_results = load_results(PPO_PATH)

    sft_indices = set(sft_results.keys())
    ppo_indices = set(ppo_results.keys())

    common_indices = sorted(
        sft_indices & ppo_indices
    )

    if not common_indices:
        raise ValueError(
            "No common prompt indices found between SFT and PPO results."
        )

    if sft_indices != ppo_indices:
        missing_from_ppo = sorted(
            sft_indices - ppo_indices
        )

        missing_from_sft = sorted(
            ppo_indices - sft_indices
        )

        print("WARNING: Prompt sets do not match.")

        if missing_from_ppo:
            print(
                f"Missing from PPO: {missing_from_ppo}"
            )

        if missing_from_sft:
            print(
                f"Missing from SFT: {missing_from_sft}"
            )

        print()

    comparison_results = []

    sft_rewards = []
    ppo_rewards = []
    deltas = []

    improved = []
    worsened = []
    unchanged = []

    for index in common_indices:
        sft = sft_results[index]
        ppo = ppo_results[index]

        sft_reward = float(sft["reward"])
        ppo_reward = float(ppo["reward"])

        delta = ppo_reward - sft_reward

        result = {
            "index": index,
            "prompt": sft["prompt"],
            "sft_response": sft["response"],
            "ppo_response": ppo["response"],
            "sft_reward": sft_reward,
            "ppo_reward": ppo_reward,
            "reward_delta": delta,
        }

        comparison_results.append(result)

        sft_rewards.append(sft_reward)
        ppo_rewards.append(ppo_reward)
        deltas.append(delta)

        if delta > IMPROVEMENT_THRESHOLD:
            improved.append(result)

        elif delta < -IMPROVEMENT_THRESHOLD:
            worsened.append(result)

        else:
            unchanged.append(result)

    num_prompts = len(comparison_results)

    mean_sft_reward = (
        sum(sft_rewards) / num_prompts
    )

    mean_ppo_reward = (
        sum(ppo_rewards) / num_prompts
    )

    mean_delta = (
        sum(deltas) / num_prompts
    )

    sorted_deltas = sorted(deltas)

    median_delta = sorted_deltas[
        len(sorted_deltas) // 2
    ]

    if len(sorted_deltas) % 2 == 0:
        middle = len(sorted_deltas) // 2

        median_delta = (
            sorted_deltas[middle - 1]
            + sorted_deltas[middle]
        ) / 2

    reward_change_pct = 0.0

    if mean_sft_reward != 0:
        reward_change_pct = (
            mean_delta
            / abs(mean_sft_reward)
            * 100
        )

    # ------------------------------------------------------------------
    # Save detailed comparison
    # ------------------------------------------------------------------

    Path(OUTPUT_PATH).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        for result in comparison_results:
            f.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"Prompts compared       : {num_prompts}")
    print(f"Mean SFT RM reward     : {mean_sft_reward:.4f}")
    print(f"Mean PPO RM reward     : {mean_ppo_reward:.4f}")
    print(f"Mean reward delta      : {mean_delta:+.4f}")
    print(f"Median reward delta    : {median_delta:+.4f}")
    print(f"Reward change          : {reward_change_pct:+.2f}%")

    print()

    print("=" * 70)
    print("PROMPT-LEVEL RESULTS")
    print("=" * 70)

    print(
        f"PPO improved          : {len(improved):3d}"
    )
    print(
        f"PPO worsened          : {len(worsened):3d}"
    )
    print(
        f"Approximately same    : {len(unchanged):3d}"
    )

    # ------------------------------------------------------------------
    # Top improvements
    # ------------------------------------------------------------------

    top_improvements = sorted(
        comparison_results,
        key=lambda x: x["reward_delta"],
        reverse=True,
    )[:10]

    print()
    print("=" * 70)
    print("TOP 10 IMPROVEMENTS")
    print("=" * 70)

    for rank, result in enumerate(
        top_improvements,
        start=1,
    ):
        print()
        print(
            f"{rank}. Index {result['index']}"
        )
        print(
            f"   SFT reward : "
            f"{result['sft_reward']:+.4f}"
        )
        print(
            f"   PPO reward : "
            f"{result['ppo_reward']:+.4f}"
        )
        print(
            f"   Delta      : "
            f"{result['reward_delta']:+.4f}"
        )
        print(
            f"   Prompt     : "
            f"{result['prompt']}"
        )

    # ------------------------------------------------------------------
    # Top regressions
    # ------------------------------------------------------------------

    top_regressions = sorted(
        comparison_results,
        key=lambda x: x["reward_delta"],
    )[:10]

    print()
    print("=" * 70)
    print("TOP 10 REGRESSIONS")
    print("=" * 70)

    for rank, result in enumerate(
        top_regressions,
        start=1,
    ):
        print()
        print(
            f"{rank}. Index {result['index']}"
        )
        print(
            f"   SFT reward : "
            f"{result['sft_reward']:+.4f}"
        )
        print(
            f"   PPO reward : "
            f"{result['ppo_reward']:+.4f}"
        )
        print(
            f"   Delta      : "
            f"{result['reward_delta']:+.4f}"
        )
        print(
            f"   Prompt     : "
            f"{result['prompt']}"
        )

    # ------------------------------------------------------------------
    # Representative side-by-side examples
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("TOP 5 IMPROVEMENT EXAMPLES")
    print("=" * 70)

    for rank, result in enumerate(
        top_improvements[:5],
        start=1,
    ):
        print()
        print("-" * 70)
        print(
            f"Example {rank} | "
            f"Index {result['index']}"
        )
        print("-" * 70)

        print()
        print(
            f"Prompt:\n"
            f"{result['prompt']}"
        )

        print()
        print(
            f"SFT "
            f"[{result['sft_reward']:+.4f}]:\n"
            f"{result['sft_response']}"
        )

        print()
        print(
            f"PPO "
            f"[{result['ppo_reward']:+.4f}]:\n"
            f"{result['ppo_response']}"
        )

    # ------------------------------------------------------------------
    # Representative regressions
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("TOP 5 REGRESSION EXAMPLES")
    print("=" * 70)

    for rank, result in enumerate(
        top_regressions[:5],
        start=1,
    ):
        print()
        print("-" * 70)
        print(
            f"Example {rank} | "
            f"Index {result['index']}"
        )
        print("-" * 70)

        print()
        print(
            f"Prompt:\n"
            f"{result['prompt']}"
        )

        print()
        print(
            f"SFT "
            f"[{result['sft_reward']:+.4f}]:\n"
            f"{result['sft_response']}"
        )

        print()
        print(
            f"PPO "
            f"[{result['ppo_reward']:+.4f}]:\n"
            f"{result['ppo_response']}"
        )

    # ------------------------------------------------------------------
    # Final output
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)

    print(
        f"Detailed results: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
