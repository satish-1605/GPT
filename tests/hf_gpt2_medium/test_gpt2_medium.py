import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():

    model_name = "gpt2-medium"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loading model on CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name
    )

    model.to("cpu")
    model.eval()

    print("\nModel loaded successfully!")
    print("Device: CPU")

    prompts = [
        "The capital of France is",
        "Artificial intelligence is",
        "Machine learning is",
    ]

    for prompt in prompts:

        print("\n" + "=" * 60)
        print("PROMPT:")
        print(prompt)
        print("=" * 60)

        inputs = tokenizer(
            prompt,
            return_tensors="pt"
        )

        with torch.no_grad():

            output = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )

        print("\nGENERATED:")
        print(generated_text)


if __name__ == "__main__":
    main()