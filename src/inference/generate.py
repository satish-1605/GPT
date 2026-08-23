from src.models.gpt import GPT2
from src.utils.config import GPTConfig
from src.training.checkpoint import load_model_checkpoint
from src.datasets.clean import clean_document
from src.tokenizer.tokenizer import BPETokenizer
from src.inference.sampling import (greedy_sampling, temperature_sampling, top_k_sampling, 
                                    top_p_sampling)

import torch

config = GPTConfig()
tokenizer = BPETokenizer.from_pretrained(config.training.load_dir)
model = GPT2(config)


model = load_model_checkpoint(path=config.training.checkpoint_path,
                             model= model,
                             device = config.training.device)

def generate(prompt,
             max_new_tokens = 100,
             sampling_strategy ="greedy_sampling",
             T=0.5,
             k=10, 
             p=0.8):
    """
    Generate text using greedy decoding.

    Args:
        prompt (str): Input prompt.
        max_new_tokens (int): Number of new tokens to generate.

    Returns:
        str: Generated text.
    """
    prompt = clean_document(prompt)   

    generated_ids = tokenizer.encode(prompt) 
    eos_token_id = tokenizer.encode("<|endoftext|>")[0]


    with torch.no_grad():
        for _ in range(max_new_tokens):

            context_ids  = generated_ids[-max_new_tokens:]

            input_ids = torch.tensor(context_ids,
                                    dtype=torch.long, 
                                    device=config.training.device).unsqueeze(0)

            
            logits = model(input_ids)
            if sampling_strategy == "greedy_sampling":
                next_token_id = greedy_sampling(logits).item()

            elif sampling_strategy == "temp_sampling":
                next_token_id = temperature_sampling(logits, T).item()

            elif sampling_strategy == "top_k_sampling":
                next_token_id = top_k_sampling(logits, k).item()

            elif sampling_strategy == "top_p_sampling":
                next_token_id = top_p_sampling(logits, p).item()
            else:
                raise ValueError("Sampling strategy is not found")

            generated_ids.append(next_token_id)


            if eos_token_id is not None and next_token_id == eos_token_id:
                break

            
    generated_text = tokenizer.decode(generated_ids)
    return generated_text

if __name__ == "__main__":
    text = generate("Once upon a time there", sampling_strategy="greedy_sampling")
    print(text)