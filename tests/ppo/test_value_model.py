from transformers import AutoTokenizer

from src.alignment.ppo.value_model import ValueModel
from src.alignment.sft.sft_config import SFTConfig

def main():
    config = SFTConfig()
    checkpoint_path = "artifacts/sft/sft_best.pt"
    device = config.device
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    tokenizer.pad_token = tokenizer.eos_token

    v_model = ValueModel(config.model_name, checkpoint_path)

    v_model.to(device)
    v_model.train()

    print("Value Model loaded successfully.")


    #test

    prompt = (
        "### Instruction:\n"
        "Explain machine learning in simple terms.\n\n"
    )

    inputs = tokenizer(prompt, return_tensors= "pt", padding=True, 
                       truncation=True, max_length = 1024)

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    values = v_model(input_ids, attention_mask)

    print(f"Input shape: {input_ids.shape}")
    print(f"Value shape: {values.shape}")
    print(f"Values: {values}")

    loss = values.mean()
    loss.backward()

    print(
    "Value head gradient:",
    v_model.value_head.weight.grad is not None)

    print("Gradient test passed.")

    print("\nSmoke test completed successfully.")

if __name__ == "__main__":
    main()



