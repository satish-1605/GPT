import torch

from src.alignment.sft.sft_config import SFTConfig
from src.utils.config import GPTConfig


def main():

    config = SFTConfig()
    gpt_config = GPTConfig()

    checkpoint_path = "artifacts/sft/sft_best.pt"

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    state_dict = checkpoint["model_state_dict"]

    print("=" * 70)
    print("SFT CHECKPOINT INSPECTION")
    print("=" * 70)

    print(f"Checkpoint : {checkpoint_path}")
    print(
        f"Global step : "
        f"{checkpoint.get('global_step')}"
    )

    print(
        f"Best val loss : "
        f"{checkpoint.get('best_val_loss')}"
    )

    print("\nGPT CONFIG")
    print(f"Vocab size     : {gpt_config.model.vocab_size}")
    print(f"Context length : {gpt_config.model.context_length}")
    print(f"d_model        : {gpt_config.model.d_model}")
    print(f"Layers         : {gpt_config.model.num_layers}")
    print(f"Heads          : {gpt_config.model.num_heads}")

    print("\nCHECKPOINT TENSORS")

    for name, tensor in list(state_dict.items())[:10]:
        print(
            f"{name:50s} "
            f"{tuple(tensor.shape)}"
        )

    print("\nTotal tensors:", len(state_dict))

    print("=" * 70)


if __name__ == "__main__":
    main()