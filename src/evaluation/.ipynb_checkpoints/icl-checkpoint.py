import itertools
import random
from collections import defaultdict

import torch

from src.models.gpt import GPT
from src.utils.config import GPTConfig
from src.tokenizer.tokenizer import BPETokenizer


# ============================================================
# Configuration
# ============================================================

CHECKPOINT_PATH = (
    "artifacts/gpt3mini_checkpoints/best_checkpoint.pt"
)

TOKENIZER_PATH = "artifacts/tokenizer"

MAX_CONTEXT_LENGTH = 512


# ============================================================
# Tasks
# ============================================================

TASKS = {

    # --------------------------------------------------------
    # Task 1: Sentiment Classification
    # --------------------------------------------------------

    "sentiment": {

        "name": "Sentiment Classification",

        "examples": [

            ("I loved this movie.", "positive"),
            ("This was terrible.", "negative"),
            ("The experience was amazing.", "positive"),
            ("I hated the product.", "negative"),
            ("The service was excellent.", "positive"),

        ],

        "queries": [

            ("The movie was fantastic.", "positive"),
            ("The product was awful.", "negative"),
            ("I really enjoyed the experience.", "positive"),
            ("This was one of the worst things ever.", "negative"),
            ("The service was wonderful.", "positive"),

            ("The movie was boring.", "negative"),
            ("I absolutely loved it.", "positive"),
            ("The product was disappointing.", "negative"),
            ("It was an excellent experience.", "positive"),
            ("I hated every minute of it.", "negative"),

        ],
    },


    # --------------------------------------------------------
    # Task 2: Topic Classification
    # --------------------------------------------------------

    "topic": {

        "name": "Topic Classification",

        "examples": [

            ("The team won the championship.", "sports"),
            ("The stock market rose today.", "finance"),
            ("The new processor is faster.", "technology"),
            ("The player scored three goals.", "sports"),
            ("The company reported higher profits.", "finance"),

        ],

        "queries": [

            ("The football team won the match.", "sports"),
            ("The company reported record earnings.", "finance"),
            ("The new smartphone uses an advanced chip.", "technology"),
            ("The player scored two goals.", "sports"),
            ("The stock price increased sharply.", "finance"),

            ("The computer has a faster processor.", "technology"),
            ("The team reached the final.", "sports"),
            ("The bank announced higher profits.", "finance"),
            ("The new software uses artificial intelligence.", "technology"),
            ("The athlete broke the world record.", "sports"),

        ],
    },


    # --------------------------------------------------------
    # Task 3: Synthetic Mapping
    # --------------------------------------------------------

    "mapping": {

        "name": "Synthetic Symbol Mapping",

        "examples": [

            ("Input: A\nOutput:", "1"),
            ("Input: B\nOutput:", "2"),
            ("Input: C\nOutput:", "3"),
            ("Input: A\nOutput:", "1"),
            ("Input: B\nOutput:", "2"),

        ],

        "queries": [

            ("Input: A\nOutput:", "1"),
            ("Input: B\nOutput:", "2"),
            ("Input: C\nOutput:", "3"),
            ("Input: A\nOutput:", "1"),
            ("Input: B\nOutput:", "2"),

            ("Input: C\nOutput:", "3"),
            ("Input: A\nOutput:", "1"),
            ("Input: B\nOutput:", "2"),
            ("Input: C\nOutput:", "3"),
            ("Input: A\nOutput:", "1"),

        ],
    },
}


# ============================================================
# Model Loading
# ============================================================

def load_model():

    config = GPTConfig()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = GPT(config).to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    else:

        model.load_state_dict(checkpoint)

    model.eval()

    tokenizer = BPETokenizer.from_pretrained(
        TOKENIZER_PATH
    )

    return model, tokenizer, device


# ============================================================
# Token Log Probability
# ============================================================

@torch.no_grad()
def sequence_log_probability(
    model,
    tokenizer,
    device,
    prompt,
    continuation,
):
    """
    Calculate the log probability of a continuation
    given a prompt.
    """

    full_text = prompt + continuation

    prompt_ids = tokenizer.encode(prompt)

    full_ids = tokenizer.encode(
        full_text
    )

    if len(full_ids) > MAX_CONTEXT_LENGTH:

        full_ids = full_ids[
            -MAX_CONTEXT_LENGTH:
        ]

    input_ids = torch.tensor(
        [full_ids],
        dtype=torch.long,
        device=device,
    )

    logits = model(input_ids)

    log_probs = torch.log_softmax(
        logits,
        dim=-1,
    )

    total_log_prob = 0.0

    continuation_ids = tokenizer.encode(
        continuation
    )

    start = (
        len(full_ids)
        - len(continuation_ids)
    )

    for i, token_id in enumerate(
        continuation_ids
    ):

        position = start + i

        if position == 0:
            continue

        token_log_prob = log_probs[
            0,
            position - 1,
            token_id,
        ]

        total_log_prob += (
            token_log_prob.item()
        )

    return total_log_prob


# ============================================================
# Build Prompt
# ============================================================

def build_prompt(
    examples,
    query,
    k,
):

    prompt = ""

    for example, label in examples[:k]:

        prompt += (
            f"{example}"
            f" {label}\n\n"
        )

    prompt += query

    return prompt


# ============================================================
# Evaluate One Example
# ============================================================

def predict(
    model,
    tokenizer,
    device,
    prompt,
    labels,
):

    scores = {}

    for label in labels:

        score = sequence_log_probability(
            model,
            tokenizer,
            device,
            prompt,
            " " + label,
        )

        scores[label] = score

    prediction = max(
        scores,
        key=scores.get,
    )

    return prediction, scores


# ============================================================
# Evaluate K-Shot
# ============================================================

def evaluate_k_shot(
    model,
    tokenizer,
    device,
    task,
    k,
):

    examples = task["examples"]
    queries = task["queries"]

    labels = sorted(
        list(
            set(
                label
                for _, label in examples
            )
        )
    )

    correct = 0

    first_label_predictions = 0

    print("\n" + "=" * 70)

    if k == 0:
        print("0-SHOT")
    else:
        print(f"{k}-SHOT")

    print("=" * 70)

    for query, expected in queries:

        prompt = build_prompt(
            examples,
            query,
            k,
        )

        prediction, scores = predict(
            model,
            tokenizer,
            device,
            prompt,
            labels,
        )

        if prediction == expected:
            correct += 1

        if prediction == labels[0]:
            first_label_predictions += 1

        print("\nPrompt:")
        print(prompt)

        print("\nLabel scores:")

        for label, score in scores.items():

            print(
                f"  {label:<12}"
                f"log P = {score:.4f}"
            )

        print("\nPrediction:")
        print(prediction)

        print("Expected:")
        print(expected)

        print(
            "Result: "
            f"{'✓' if prediction == expected else '✗'}"
        )

    accuracy = (
        correct /
        len(queries)
    )

    prediction_rate = (
        first_label_predictions /
        len(queries)
    )

    print("\n" + "-" * 70)

    print(
        f"{k}-shot Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"First-label prediction rate: "
        f"{prediction_rate * 100:.2f}%"
    )

    print("-" * 70)

    return accuracy


# ============================================================
# Evaluate Task
# ============================================================

def evaluate_task(
    model,
    tokenizer,
    device,
    task_name,
    task,
):

    print("\n\n")

    print("#" * 70)

    print(
        f"TASK: {task['name']}"
    )

    print("#" * 70)

    results = {}

    for k in [0, 1, 3, 5]:

        accuracy = evaluate_k_shot(
            model,
            tokenizer,
            device,
            task,
            k,
        )

        results[k] = accuracy

    return results


# ============================================================
# Balanced Permutation ICL
# ============================================================

def get_single_token_id(
    tokenizer,
    text,
):
    """
    Verify that a label is represented by exactly
    one token.
    """

    token_ids = tokenizer.encode(
        text
    )

    if len(token_ids) != 1:

        raise ValueError(
            f"Expected '{text}' to be exactly "
            f"one token, but tokenizer returned "
            f"{token_ids}"
        )

    return token_ids[0]


# ============================================================
# Score Next Token
# ============================================================

@torch.no_grad()
def score_next_token(
    model,
    tokenizer,
    device,
    prompt,
    label_token_ids,
):
    """
    Score candidate labels using the probability
    of the next token after the prompt.
    """

    prompt_ids = tokenizer.encode(
        prompt
    )

    if len(prompt_ids) == 0:

        raise ValueError(
            "Prompt produced zero tokens."
        )

    if len(prompt_ids) > MAX_CONTEXT_LENGTH:

        prompt_ids = prompt_ids[
            -MAX_CONTEXT_LENGTH:
        ]

    input_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device,
    )

    logits = model(
        input_ids
    )

    next_token_logits = logits[
        0,
        -1,
    ]

    log_probs = torch.log_softmax(
        next_token_logits,
        dim=-1,
    )

    scores = {}

    for label, token_id in (
        label_token_ids.items()
    ):

        scores[label] = (
            log_probs[token_id].item()
        )

    return scores


# ============================================================
# Balanced Permutation-Invariant Benchmark
# ============================================================

def evaluate_balanced_permutation_icl(
    model,
    tokenizer,
    device,
    seed=42,
    repeats_per_symbol=10,
):
    """
    Balanced permutation-invariant ICL benchmark.

    Six possible A/B/C -> 1/2/3 mappings are tested.

    For every permutation:

        - A, B and C are queried equally
        - demonstration order is randomized
        - query order is randomized
        - each permutation receives equal trials

    Reports:

        - overall accuracy
        - balanced accuracy
        - per-label accuracy
        - first-label prediction rate
        - per-permutation accuracy
        - mean/std/min/max permutation accuracy
    """

    print("\n")

    print("#" * 70)

    print(
        "TASK: Balanced Permutation-Invariant "
        "Synthetic ICL"
    )

    print("#" * 70)

    # ========================================================
    # Configuration
    # ========================================================

    symbols = [
        "A",
        "B",
        "C",
    ]

    labels = [
        "1",
        "2",
        "3",
    ]

    rng = random.Random(
        seed
    )

    permutations = list(
        itertools.permutations(
            labels
        )
    )

    # ========================================================
    # Tokenization Check
    # ========================================================

    print("\n" + "=" * 70)
    print("TOKENIZATION CHECK")
    print("=" * 70)

    label_token_ids = {}

    for label in labels:

        token_ids = tokenizer.encode(
            " " + label
        )

        print(
            f"Label '{label}' -> "
            f"tokens {token_ids}"
        )

        if len(token_ids) != 1:

            raise ValueError(
                f"Label '{label}' must be represented "
                f"by exactly one token. "
                f"Got {token_ids}"
            )

        label_token_ids[label] = (
            token_ids[0]
        )

    print(
        "\nTokenizer check passed."
    )

    # ========================================================
    # Global Statistics
    # ========================================================

    total_correct = 0
    total_predictions = 0

    first_label_predictions = 0

    label_correct = defaultdict(int)
    label_total = defaultdict(int)

    permutation_accuracies = []

    # ========================================================
    # Evaluate All Six Permutations
    # ========================================================

    for permutation_index, permutation in enumerate(
        permutations,
        start=1,
    ):

        mapping = dict(
            zip(
                symbols,
                permutation,
            )
        )

        print("\n")

        print("=" * 70)

        print(
            f"PERMUTATION "
            f"{permutation_index}/6"
        )

        print("=" * 70)

        print(
            f"Mapping: "
            f"A->{mapping['A']} "
            f"B->{mapping['B']} "
            f"C->{mapping['C']}"
        )

        # ----------------------------------------------------
        # Demonstrations
        # ----------------------------------------------------

        demonstrations = [
            (
                symbol,
                mapping[symbol],
            )
            for symbol in symbols
        ]

        permutation_correct = 0
        permutation_total = 0

        # ----------------------------------------------------
        # Balanced Query Set
        # ----------------------------------------------------

        query_symbols = []

        for symbol in symbols:

            query_symbols.extend(
                [symbol] *
                repeats_per_symbol
            )

        # Randomize query order
        rng.shuffle(
            query_symbols
        )

        # ----------------------------------------------------
        # Evaluate Queries
        # ----------------------------------------------------

        for trial_index, query_symbol in enumerate(
            query_symbols,
            start=1,
        ):

            expected = mapping[
                query_symbol
            ]

            # -----------------------------------------------
            # Randomize demonstration order
            # -----------------------------------------------

            shuffled_demonstrations = (
                demonstrations.copy()
            )

            rng.shuffle(
                shuffled_demonstrations
            )

            # -----------------------------------------------
            # Build Prompt
            # -----------------------------------------------

            prompt = ""

            for symbol, label in (
                shuffled_demonstrations
            ):

                prompt += (
                    f"Input: {symbol}\n"
                    f"Output: {label}\n\n"
                )

            prompt += (
                f"Input: {query_symbol}\n"
                f"Output:"
            )

            # -----------------------------------------------
            # Score Candidate Labels
            # -----------------------------------------------

            scores = score_next_token(
                model,
                tokenizer,
                device,
                prompt,
                label_token_ids,
            )

            prediction = max(
                scores,
                key=scores.get,
            )

            # -----------------------------------------------
            # Statistics
            # -----------------------------------------------

            is_correct = (
                prediction ==
                expected
            )

            if is_correct:

                permutation_correct += 1
                total_correct += 1

                label_correct[
                    expected
                ] += 1

            label_total[
                expected
            ] += 1

            permutation_total += 1
            total_predictions += 1

            if prediction == labels[0]:

                first_label_predictions += 1

            # -----------------------------------------------
            # Output
            # -----------------------------------------------

            print(
                "\n" + "-" * 70
            )

            print(
                f"Trial {trial_index}/"
                f"{len(query_symbols)}"
            )

            print(
                f"Query:    "
                f"{query_symbol}"
            )

            print(
                f"Expected: "
                f"{expected}"
            )

            print(
                "Demonstration order: "
                + " ".join(
                    symbol
                    for symbol, _ in
                    shuffled_demonstrations
                )
            )

            print("\nPrompt:")
            print(prompt)

            print("Label scores:")

            for label in labels:

                print(
                    f"  {label:<10}"
                    f"log P = "
                    f"{scores[label]:.4f}"
                )

            print(
                f"\nPrediction: "
                f"{prediction}"
            )

            print(
                "Result: "
                f"{'✓' if is_correct else '✗'}"
            )

        # ----------------------------------------------------
        # Permutation Result
        # ----------------------------------------------------

        permutation_accuracy = (
            permutation_correct /
            permutation_total
        )

        permutation_accuracies.append(
            permutation_accuracy
        )

        print("\n" + "=" * 70)

        print(
            f"Permutation "
            f"{permutation_index} "
            f"Accuracy: "
            f"{permutation_accuracy * 100:.2f}%"
        )

        print("=" * 70)

    # ========================================================
    # Overall Accuracy
    # ========================================================

    overall_accuracy = (
        total_correct /
        total_predictions
    )

    # ========================================================
    # Per-Label Accuracy
    # ========================================================

    per_label_accuracy = {}

    for label in labels:

        if label_total[label] == 0:

            per_label_accuracy[
                label
            ] = 0.0

        else:

            per_label_accuracy[
                label
            ] = (
                label_correct[label] /
                label_total[label]
            )

    # ========================================================
    # Balanced Accuracy
    # ========================================================

    balanced_accuracy = (
        sum(
            per_label_accuracy.values()
        )
        /
        len(labels)
    )

    # ========================================================
    # First-Label Prediction Rate
    # ========================================================

    first_label_rate = (
        first_label_predictions /
        total_predictions
    )

    # ========================================================
    # Permutation Statistics
    # ========================================================

    permutation_tensor = torch.tensor(
        permutation_accuracies,
        dtype=torch.float32,
    )

    permutation_mean = (
        permutation_tensor.mean().item()
    )

    permutation_std = (
        permutation_tensor.std(
            unbiased=False
        ).item()
    )

    permutation_min = (
        permutation_tensor.min().item()
    )

    permutation_max = (
        permutation_tensor.max().item()
    )

    # ========================================================
    # Final Report
    # ========================================================

    print("\n\n")

    print("#" * 70)

    print(
        "BALANCED PERMUTATION ICL RESULTS"
    )

    print("#" * 70)

    print(
        f"\nOverall Accuracy:              "
        f"{overall_accuracy * 100:.2f}%"
    )

    print(
        f"Random Baseline:               "
        f"{100 / 3:.2f}%"
    )

    print(
        f"Balanced Accuracy:             "
        f"{balanced_accuracy * 100:.2f}%"
    )

    print(
        f"First-label Prediction Rate:   "
        f"{first_label_rate * 100:.2f}%"
    )

    print(
        f"Expected Uniform Rate:         "
        f"{100 / 3:.2f}%"
    )

    # ========================================================
    # Per-Label Accuracy
    # ========================================================

    print("\n" + "-" * 70)

    print(
        "PER-LABEL ACCURACY"
    )

    print("-" * 70)

    for label in labels:

        print(
            f"Label {label}: "
            f"{per_label_accuracy[label] * 100:.2f}% "
            f"("
            f"{label_correct[label]}/"
            f"{label_total[label]}"
            f")"
        )

    # ========================================================
    # Per-Permutation Accuracy
    # ========================================================

    print("\n" + "-" * 70)

    print(
        "PERMUTATION ACCURACY"
    )

    print("-" * 70)

    for index, accuracy in enumerate(
        permutation_accuracies,
        start=1,
    ):

        permutation = permutations[
            index - 1
        ]

        mapping = dict(
            zip(
                symbols,
                permutation,
            )
        )

        print(
            f"P{index}: "
            f"{accuracy * 100:.2f}%   "
            f"("
            f"A->{mapping['A']} "
            f"B->{mapping['B']} "
            f"C->{mapping['C']}"
            f")"
        )

    # ========================================================
    # Permutation Distribution
    # ========================================================

    print("\n" + "-" * 70)

    print(
        "PERMUTATION DISTRIBUTION"
    )

    print("-" * 70)

    print(
        f"Mean: "
        f"{permutation_mean * 100:.2f}%"
    )

    print(
        f"Std:  "
        f"{permutation_std * 100:.2f}%"
    )

    print(
        f"Min:  "
        f"{permutation_min * 100:.2f}%"
    )

    print(
        f"Max:  "
        f"{permutation_max * 100:.2f}%"
    )

    print("\n" + "#" * 70)

    return {
        "accuracy": overall_accuracy,
        "balanced_accuracy": balanced_accuracy,
        "first_label_rate": first_label_rate,
        "per_label_accuracy": per_label_accuracy,
        "permutation_results": permutation_accuracies,
        "permutation_mean": permutation_mean,
        "permutation_std": permutation_std,
        "permutation_min": permutation_min,
        "permutation_max": permutation_max,
    }


# ============================================================
# Main
# ============================================================

def main():

    # ========================================================
    # Load Model
    # ========================================================

    model, tokenizer, device = load_model()

    print("=" * 70)

    print(
        "GPT-3 MINI — STRENGTHENED ICL BENCHMARK"
    )

    print("=" * 70)

    print(
        f"Device: {device}"
    )

    # ========================================================
    # Standard ICL Benchmark
    # ========================================================

    all_results = {}

    for task_name, task in TASKS.items():

        results = evaluate_task(
            model,
            tokenizer,
            device,
            task_name,
            task,
        )

        all_results[
            task_name
        ] = results

    # ========================================================
    # Final Standard ICL Summary
    # ========================================================

    print("\n\n")

    print("=" * 70)

    print(
        "FINAL ICL BENCHMARK RESULTS"
    )

    print("=" * 70)

    print(
        f"{'Task':<25}"
        f"{'0-shot':>10}"
        f"{'1-shot':>10}"
        f"{'3-shot':>10}"
        f"{'5-shot':>10}"
    )

    print("-" * 70)

    for task_name, results in (
        all_results.items()
    ):

        print(
            f"{TASKS[task_name]['name']:<25}"
            f"{results[0] * 100:>9.2f}%"
            f"{results[1] * 100:>9.2f}%"
            f"{results[3] * 100:>9.2f}%"
            f"{results[5] * 100:>9.2f}%"
        )

    print("=" * 70)

    # ========================================================
    # Balanced Permutation-Invariant ICL
    # ========================================================

    permutation_results = (
        evaluate_balanced_permutation_icl(
            model,
            tokenizer,
            device,
            seed=42,
            repeats_per_symbol=10,
        )
    )

    # ========================================================
    # Final Combined Summary
    # ========================================================

    print("\n\n")

    print("=" * 70)

    print(
        "GPT-3 MINI — FINAL BENCHMARK SUMMARY"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Standard ICL
    # --------------------------------------------------------

    print("\nStandard ICL:")

    for task_name, results in (
        all_results.items()
    ):

        print(
            f"  "
            f"{TASKS[task_name]['name']:<25}"
            f"0-shot="
            f"{results[0] * 100:.2f}%  "
            f"1-shot="
            f"{results[1] * 100:.2f}%  "
            f"3-shot="
            f"{results[3] * 100:.2f}%  "
            f"5-shot="
            f"{results[5] * 100:.2f}%"
        )

    # --------------------------------------------------------
    # Balanced Permutation ICL
    # --------------------------------------------------------

    print(
        "\nBalanced Permutation ICL:"
    )

    print(
        f"  Overall Accuracy:      "
        f"{permutation_results['accuracy'] * 100:.2f}%"
    )

    print(
        f"  Balanced Accuracy:     "
        f"{permutation_results['balanced_accuracy'] * 100:.2f}%"
    )

    print(
        f"  First-label Rate:      "
        f"{permutation_results['first_label_rate'] * 100:.2f}%"
    )

    print(
        f"  Permutation Mean:      "
        f"{permutation_results['permutation_mean'] * 100:.2f}%"
    )

    print(
        f"  Permutation Std:       "
        f"{permutation_results['permutation_std'] * 100:.2f}%"
    )

    print(
        f"  Permutation Min:       "
        f"{permutation_results['permutation_min'] * 100:.2f}%"
    )

    print(
        f"  Permutation Max:       "
        f"{permutation_results['permutation_max'] * 100:.2f}%"
    )

    print("=" * 70)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()