import torch

from src.alignment.sft.sft_config import SFTConfig
from src.tokenizer.hf_tokenizer import HFTokenizer
from src.models.gpt import GPT
from src.utils.config import GPTConfig

EOS_TOKEN_ID = 0
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.7
TOP_K = 50

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

@torch.no_grad()
def generate(model, tokenizer, prompt, config,
             max_new_tokens = MAX_NEW_TOKENS, 
             temperature=TEMPERATURE, 
             top_k=TOP_K,
             ):
    model.eval()

    input_ids = tokenizer.encode(prompt)


    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(config.training.device)



    generated_ids = input_ids

    for _ in range(max_new_tokens):
        context = generated_ids[:,-config.model.context_length:]

        logits = model(context)
        logits = logits[:, -1, :]

        logits = logits / temperature

        if top_k is not None:
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < values[:, [-1]]] = float("-inf")

        probabilities = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)
        generated_ids = torch.cat([generated_ids, next_token], dim=1)

        if next_token.item() == EOS_TOKEN_ID:
            break

    response_ids = generated_ids[0, input_ids.size(1):].tolist()
    response = tokenizer.decode(response_ids)
    return response

def main():
    print("=" * 70)
    print("SFT INFERENCE")
    print("=" * 70)

    sft_config = SFTConfig()
    tokenizer = HFTokenizer(sft_config.tokenizer_path)

    print(f"Tokenizer loaded from: {sft_config.tokenizer_path}")

    gpt_config = GPTConfig()
    model = GPT(gpt_config).to(gpt_config.training.device)

    checkpoint_path = ("artifacts/sft/sft_best.pt")
    checkpoint = torch.load(checkpoint_path, map_location=sft_config.device, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    print(f"SFT checkpoint loaded: {checkpoint_path}")
    print(f"Best validation loss: {checkpoint.get('best_val_loss', 'N/A')}")
    print(f"Device: {sft_config.device}")
    print("=" * 70)

    instruction = ("Why can camels survive for long without water?")
    input = ""
    prompt = format_prompt(instruction, input)


    print("\nINSTRUCTION:")
    print(instruction)

    print("\nGENERATING...\n")

    response = generate(model, tokenizer, prompt, gpt_config)

    print("RESPONSE:")
    print(response)

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()



