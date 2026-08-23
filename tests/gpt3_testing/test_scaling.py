from src.models.gpt import GPT
from src.utils.config import GPT3_MODELS
from src.utils.model_utils import count_parameters_millions


def test_scaling():
    parameter_counts = {}

    # First, get parameter counts
    for name, config_fn in GPT3_MODELS.items():
        config = config_fn()
        model = GPT(config)

        parameter_counts[name] = count_parameters_millions(model)

    # Use GPT-2 as the baseline
    baseline = parameter_counts["gpt2_baseline"]

    # Print absolute size + relative scale
    for name, params in parameter_counts.items():
        scale = params / baseline

        print(
            f"{name}: {params:.2f}M | {scale:.2f}×"
        )


if __name__ == "__main__":
    test_scaling()