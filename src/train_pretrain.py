
from src.models.gpt import GPT
from src.utils.config import GPTConfig

from src.training.optimizer import create_optimizer

from src.datasets.data_pipeline import get_train_val_loaders
from src.trainer.trainer import train_one_epoch
from src.training.validate import validate
from src.utils.checkpoint import save_checkpoint

def main():
    config = GPTConfig()
    device = config.device

    model = GPT(config).to(device)

    optimizer = create_optimizer(model=model, 
                                 learning_rate=config.learning_rate)
    
    train_loader, val_loader = get_train_val_loaders(config)
    
    best_val_loss = float("inf")
    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(loader=train_loader,
                                     model=model,
                                     optimizer=optimizer,
                                     device=device)

        val_loss = validate(loader=val_loader,
                            model=model,
                            device = device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(epoch=epoch, 
                                        model=model,
                                        optimizer=optimizer,
                                        train_loss=train_loss,
                                        val_loss=val_loss,
                                        filepath=config.checkpoint_path,)

        print(
        f"Epoch [{epoch}/{config.epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Val Loss: {val_loss:.4f}"
    )


if __name__ == "__main__":
    main()
        


    
    





