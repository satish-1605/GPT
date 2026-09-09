import torch
from transformers import AutoTokenizer

from src.alignment.reward.reward_model import GPT2RewardModel
from src.alignment.sft.sft_config import SFTConfig

def main():
    config = SFTConfig()
    checkpoint_path = "artifacts/sft/sft_best.pt"
    device = config.device
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2RewardModel(config.model_name, checkpoint_path)

    model.to(device)
    model.train()

    print("Reward Model loaded successfully.")


    #test

    prompt = (
        "### Instruction:\n"
        "Explain machine learning in simple terms.\n\n"
        "### Response:\n"
        "Machine learning allows computers to learn patterns "
        "from data without being explicitly programmed for "
        "every task."
    )

    inputs = tokenizer(prompt, return_tensors= "pt", padding=True, 
                       truncation=True, max_length = True)

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    reward = model(input_ids, attention_mask)

    print(f"Input shape: {input_ids.shape}")
    print(f"Reward shape: {reward.shape}")
    print(f"Reward: {reward.item():.4f}")

    loss = reward.mean()
    loss.backward()

    print("Gradient test passed.")

    print("\nSmoke test completed successfully.")

if __name__ == "__main__":
    main()



