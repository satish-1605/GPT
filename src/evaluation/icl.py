import torch

from src.utils.config import GPTConfig
from src.models.gpt import GPT
from src.tokenizer.tokenizer import BPETokenizer

from src.evaluation.prompts import CLASSIFICATION_DATA, PATTERN_DATA, TEXT_ICL_DATA

def load_model():
    config = GPTConfig()

    device = config.training.device

    tokenizer = BPETokenizer.from_pretrained(config.training.load_dir)
    model = GPT(config).to(device)

    checkpoint = torch.load(config.training.checkpoint_path, 
                            map_location=device,
                            weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, tokenizer, device

def generate(model, tokenizer, prompt, device, max_new_tokens=100, temprature=1.0, top_k=50):
    token_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)

            logits = logits[:, -1, :]
            if temprature > 0:
                logits = logits / temprature

            if top_k is not None:
                values, _ = torch.topk(logits, top_k)
                min_value = values[:, -1].unsqueeze(-1)
                logits[logits < min_value] = float("-inf")

            probabilities = torch.softmax(logits, dim=-1)

            next_token = torch.multinomial(probabilities, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)


    return tokenizer.decode(input_ids[0].tolist())

def build_icl_prompt(examples, query, shots=0):
    prompt = ""

    if shots == 0:
        prompt = (
            "Classify the sentiment as positive, negative, or neutral.\n\n"
        )

    else:
        prompt = (
            "Classify the sentiment as positive, negative, or neutral.\n\n"
        )

        for example in examples[:shots]:
            prompt += f"Text: {example['text']}\n"
            prompt += f"Sentiment: {example['label']}\n\n"

    prompt += f"Text: {query['text']}\n"
    prompt += "Sentiment:"

    return prompt


def build_completions_prompt(examples, query, shots=0):
    prompt = ""

    for example in examples[:shots]:
        prompt += example["input"] + example["target"] + "\n\n"

    prompt += query['input']
    return prompt

#---------------------------------------------------------------------

def main():
    model, tokenizer, device = load_model()
    query = TEXT_ICL_DATA[5]

    for i in [0,1,3,5]:
        prompt = build_completions_prompt(
            TEXT_ICL_DATA,
            query,
            shots=i
        )

        output = generate(
            model,
            tokenizer,
            prompt,
            device,
            max_new_tokens=5
        )

        print(f"\n{'=' * 40}")
        print(f"{i}-SHOT")
        print(output)


if __name__ == "__main__":
    main()





