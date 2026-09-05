import torch
from src.losses.loss import calculate_loss


def validate(loader, model, device):
    
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for input_ids, target_ids in loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            logits = model(input_ids)
            loss = calculate_loss(logits, target_ids)    
    
            total_loss += loss.item()
            num_batches += 1
    val_loss = total_loss/num_batches
    return val_loss 


from src.models.gpt import GPT
from src.utils.config import GPTConfig

config = GPTConfig()
from src.datasets.data_pipeline import get_train_val_test_loaders

_, val_loader, _ = get_train_val_test_loaders(config)
if __name__ == "__main__":

    fresh_model = GPT(config).to("cpu")

    fresh_val_loss = validate(
        loader=val_loader,
        model=fresh_model,
        device="cpu"
    )

    print("Fresh model validation loss:", fresh_val_loss)