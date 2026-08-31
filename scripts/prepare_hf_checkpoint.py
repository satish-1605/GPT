import json
import torch
from pathlib import Path


SOURCE = Path(
    "artifacts/gpt3large_checkpoints/best_checkpoint.pt"
)

OUTPUT_DIR = Path(
    "artifacts/hf/gpt-300m-base"
)

MODEL_FILE = OUTPUT_DIR / "model.pt"
CONFIG_FILE = OUTPUT_DIR / "config.json"


def config_to_dict(config):
    """
    Convert checkpoint config into a JSON-serializable dictionary.
    """

    if isinstance(config, dict):
        return config

    if hasattr(config, "__dict__"):
        return {
            key: value
            for key, value in config.__dict__.items()
            if not key.startswith("_")
        }

    raise TypeError(
        f"Unsupported config type: {type(config)}"
    )

def main():

    print("=" * 60)
    print("Preparing portable HF checkpoint")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Load original checkpoint
    # --------------------------------------------------

    print(f"\nLoading: {SOURCE}")

    checkpoint = torch.load(
        SOURCE,
        map_location="cpu",
        weights_only=False
    )

    print(
        f"Original keys: {list(checkpoint.keys())}"
    )

    # --------------------------------------------------
    # 2. Extract model weights
    # --------------------------------------------------

    model_state_dict = checkpoint["model_state_dict"]

    print(
        f"\nModel tensors: "
        f"{len(model_state_dict):,}"
    )

    parameter_count = sum(
        tensor.numel()
        for tensor in model_state_dict.values()
    )

    print(
        f"Parameters: "
        f"{parameter_count:,}"
    )

    # --------------------------------------------------
    # 3. Save model-only checkpoint
    # --------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        model_state_dict,
        MODEL_FILE
    )

    print(
        f"\nModel saved to:\n"
        f"{MODEL_FILE}"
    )

    # --------------------------------------------------
    # 4. Convert config to JSON
    # --------------------------------------------------

    config = checkpoint["config"]

    config_dict = config_to_dict(config)

    with CONFIG_FILE.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            config_dict,
            f,
            indent=2,
            ensure_ascii=False,
            default=str
        )

    print(
        f"Config saved to:\n"
        f"{CONFIG_FILE}"
    )

    # --------------------------------------------------
    # 5. Summary
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("HF PACKAGE PREPARATION COMPLETE")
    print("=" * 60)

    print(
        f"Parameters : {parameter_count:,}"
    )

    print(
        f"Model size : "
        f"{MODEL_FILE.stat().st_size / (1024 ** 3):.2f} GB"
    )

    print(
        f"\nOutput directory:\n"
        f"{OUTPUT_DIR.resolve()}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()