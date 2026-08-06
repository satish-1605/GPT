from src.inference.load_model import load_model
from src.utils.config import GPTConfig
from src.models.gpt import GPT
from src.evaluation.metrics import compute_perplexity
from src.trainer.validate import validate
from src.datasets.data_pipeline import get_train_val_loaders


config = GPTConfig()
model = GPT(config)

model = load_model(model=model,
                   filepath=config.checkpoint_path, 
                   device= config.device)

def evaluate():
    """
    Evaluate the trained GPT model on the validation set.

    Returns:
        tuple:
            Validation loss and perplexity.
    """
    model.eval()
    _, val_loader = get_train_val_loaders(config)

    val_loss = validate(loader=val_loader,
                        model=model,
                        device=config.device)

    perplexity = compute_perplexity(loss=val_loss)
    print("=" * 50)
    print(f"Validation Loss : {val_loss:.4f}")
    print(f"Perplexity      : {perplexity:.4f}")
    print("=" * 50)

    return {
    "val_loss": val_loss,
    "perplexity": perplexity,
}


if __name__ == "__main__":
    evaluate()