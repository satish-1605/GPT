from src.training.checkpoint import load_model_checkpoint
from src.utils.config import GPTConfig
from src.models.gpt import GPT
from src.evaluation.metrics import compute_perplexity
from src.training.validate import validate
from src.datasets.data_pipeline import get_train_val_test_loaders


config = GPTConfig()
model = GPT(config)

model = load_model_checkpoint(model=model,
                   path=config.checkpoint_path, 
                   device= config.device)

def evaluate():
    """
    Evaluate the trained GPT model on the validation set.

    Returns:
        tuple:
            Validation loss and perplexity.
    """
    model.eval()
    _, _, test_loader = get_train_val_test_loaders(config)

    test_loss = validate(loader=test_loader,
                        model=model,
                        device=config.device)

    perplexity = compute_perplexity(loss=test_loss)
    print("=" * 50)
    print(f"Validation Loss : {test_loss:.4f}")
    print(f"Perplexity      : {perplexity:.4f}")
    print("=" * 50)

    return {
    "val_loss": test_loss,
    "perplexity": perplexity,
}


if __name__ == "__main__":
    evaluate()