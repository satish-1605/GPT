from src.inference.generate import generate
from src.evaluation.prompts import PROMPTS


SAMPLING_STRATEGIES = {
    "Greedy": "greedy_sampling",
    "Temperature": "temp_sampling",
    "Top-k": "top_k_sampling",
    "Top-p": "top_p_sampling",
}


def qualitative_evaluation():
    """
    Generate text using different sampling strategies
    for a fixed set of prompts.
    """

    for prompt in PROMPTS:

        print("=" * 100)
        print(f"Prompt: {prompt}\n")

        for strategy_name, strategy in SAMPLING_STRATEGIES.items():

            print(f"{strategy_name} Sampling:\n")

            response = generate(
                prompt=prompt,
                sampling_strategy=strategy,
            )

            print(response)
            print()

        print("=" * 100)
        print()


if __name__ == "__main__":
    qualitative_evaluation()