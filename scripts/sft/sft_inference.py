import torch

from src.alignment.sft.sft_config import SFTConfig
from src.tokenizer.hf_tokenizer import HFTokenizer
from src.models.gpt import GPT
from src.utils.config import GPTConfig


# ============================================================
# Constants
# ============================================================

EOS_TOKEN_ID = 0
MAX_NEW_TOKENS = 100


# ============================================================
# Prompt Formatting
# ============================================================

def format_prompt(instruction, input=""):
    if input:
        prompt = (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Input:\n"
            f"{input}\n\n"
            "### Response:\n"
        )
    else:
        prompt = (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Response:\n"
        )

    return prompt


# ============================================================
# Model Information
# ============================================================

def print_model_info(model, config, tokenizer):
    print()
    print("=" * 70)
    print("MODEL / TOKENIZER CONFIGURATION")
    print("=" * 70)

    print(f"Vocab size     : {config.model.vocab_size}")
    print(f"Context length : {config.model.context_length}")
    print(f"d_model        : {config.model.d_model}")
    print(f"Layers         : {config.model.num_layers}")
    print(f"Heads          : {config.model.num_heads}")

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(f"Model params   : {total_params:,}")

    print(
    f"Tokenizer vocab: "
    f"{tokenizer.tokenizer.get_vocab_size()}"
)

    print("=" * 70)


# ============================================================
# First Token Diagnostic
# ============================================================

def diagnose_first_token(
    model,
    tokenizer,
    input_ids,
    config
):
    """
    Inspect probability and rank of the expected
    first response token: 'Mandatory'.
    """

    with torch.no_grad():

        context = input_ids[
            :, -config.model.context_length:
        ]

        logits = model(context)

        # Prediction for the next token
        logits = logits[:, -1, :]

        probs = torch.softmax(
            logits,
            dim=-1
        )[0]

        # ----------------------------------------------------
        # Expected first response token
        # ----------------------------------------------------

        mandatory_tokens = tokenizer.encode(
            "Mandatory"
        )

        mandatory_id = mandatory_tokens[0]

        mandatory_prob = (
            probs[mandatory_id].item()
        )

        mandatory_rank = (
            (probs > probs[mandatory_id])
            .sum()
            .item()
            + 1
        )

        # ----------------------------------------------------
        # Print diagnostic
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("FIRST RESPONSE TOKEN DIAGNOSTIC")
        print("=" * 70)

        print(
            f"Mandatory token ID : {mandatory_id}"
        )

        print(
            f"Mandatory probability : "
            f"{mandatory_prob:.8f}"
        )

        print(
            f"Mandatory rank : "
            f"{mandatory_rank}"
        )

        # ----------------------------------------------------
        # Top 20 tokens
        # ----------------------------------------------------

        top_probs, top_ids = torch.topk(
            probs,
            20
        )

        print()
        print("TOP 20 FIRST RESPONSE TOKENS:")
        print("-" * 70)

        for prob, token_id in zip(
            top_probs.tolist(),
            top_ids.tolist()
        ):
            token_text = tokenizer.decode(
                [token_id]
            )

            print(
                f"{token_id:6d} | "
                f"{prob:.6f} | "
                f"{repr(token_text)}"
            )

        print("=" * 70)


# ============================================================
# Generation
# ============================================================

def generate(
    model,
    tokenizer,
    prompt,
    config,
    max_new_tokens=MAX_NEW_TOKENS,
):
    model.eval()

    # --------------------------------------------------------
    # Encode prompt
    # --------------------------------------------------------

    encoded_ids = tokenizer.encode(prompt)

    input_ids = torch.tensor(
        encoded_ids,
        dtype=torch.long
    ).unsqueeze(0).to(
        config.training.device
    )

    generated_ids = input_ids.clone()

    # --------------------------------------------------------
    # FIRST TOKEN DIAGNOSTIC
    # --------------------------------------------------------

    diagnose_first_token(
        model,
        tokenizer,
        input_ids,
        config
    )

    # --------------------------------------------------------
    # GREEDY GENERATION
    # --------------------------------------------------------

    with torch.no_grad():

        for _ in range(max_new_tokens):

            context = generated_ids[
                :, -config.model.context_length:
            ]

            logits = model(context)

            # Last token logits
            logits = logits[:, -1, :]

            # ------------------------------------------------
            # Greedy decoding
            # ------------------------------------------------

            next_token = torch.argmax(
                logits,
                dim=-1,
                keepdim=True
            )

            generated_ids = torch.cat(
                [
                    generated_ids,
                    next_token
                ],
                dim=1
            )

            # ------------------------------------------------
            # EOS
            # ------------------------------------------------

            if next_token.item() == EOS_TOKEN_ID:
                break

    # --------------------------------------------------------
    # Decode generated response only
    # --------------------------------------------------------

    response_ids = generated_ids[
        0,
        input_ids.size(1):
    ].tolist()

    response = tokenizer.decode(
        response_ids
    )

    return response


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("SFT INFERENCE + FIRST TOKEN DIAGNOSTIC")
    print("=" * 70)

    # --------------------------------------------------------
    # SFT config
    # --------------------------------------------------------

    sft_config = SFTConfig()

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    tokenizer = HFTokenizer(
        sft_config.tokenizer_path
    )

    print(
        f"Tokenizer loaded from: "
        f"{sft_config.tokenizer_path}"
    )

    # --------------------------------------------------------
    # GPT config
    # --------------------------------------------------------

    gpt_config = GPTConfig()

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = GPT(
        gpt_config
    ).to(
        gpt_config.training.device
    )

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    checkpoint_path = (
        "artifacts/sft/sft_best.pt"
    )

    print(
        f"Checkpoint: {checkpoint_path}"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=gpt_config.training.device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # --------------------------------------------------------
    # Print checkpoint information
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CHECKPOINT INFORMATION")
    print("=" * 70)

    if "global_step" in checkpoint:
        print(
            f"Global step   : "
            f"{checkpoint['global_step']}"
        )

    if "best_val_loss" in checkpoint:
        print(
            f"Best val loss : "
            f"{checkpoint['best_val_loss']}"
        )

    print("=" * 70)

    # --------------------------------------------------------
    # Model / tokenizer configuration
    # --------------------------------------------------------

    print_model_info(
        model,
        gpt_config,
        tokenizer
    )

    # --------------------------------------------------------
    # Load SFT dataset
    # --------------------------------------------------------

    from src.alignment.sft.instruction_dataset import (
        InstructionDataset
    )

    dataset = InstructionDataset(
        sft_config.train_file,
        tokenizer,
        sft_config.max_length
    )

    print()
    print(
        f"Training examples available: "
        f"{len(dataset):,}"
    )

    # --------------------------------------------------------
    # Test instruction
    # --------------------------------------------------------

    instruction = (
        "If you are traveling outside the country, "
        "categorize each of the following as either: "
        "‘Mandatory’, ‘Good to have’, ‘Least important’. "
        "Passport, Cash of the country visiting, "
        "Power bank, Book, Pen, Laptop"
    )

    expected = (
        "Mandatory: Passport\n"
        "Good to have: Cash of the country visiting, Powerbank\n"
        "Least important: Book, Pen, Laptop"
    )

    prompt = format_prompt(
        instruction
    )

    print()
    print("=" * 70)
    print("TEST INSTRUCTION")
    print("=" * 70)

    print()
    print("INSTRUCTION:")
    print(instruction)

    print()
    print("EXPECTED:")
    print(expected)

    print()
    print("PROMPT:")
    print(repr(prompt))

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("GENERATING...")
    print("=" * 70)

    response = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        config=gpt_config,
        max_new_tokens=MAX_NEW_TOKENS
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("GENERATED:")
    print("=" * 70)

    print(response)

    print()
    print("=" * 70)
    print("SFT DIAGNOSTIC COMPLETE")
    print("=" * 70)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()