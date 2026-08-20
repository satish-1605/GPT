from src.utils.config import GPTConfig
from src.models.gpt2 import GPT2
from src.datasets.data_pipeline import get_train_val_test_loaders
from src.training.optimizer import create_optimizer
from src.training.trainer import train_one_epoch
from src.training.validate import validate
from src.training.checkpoint import save_checkpoint

def main():
    config = GPTConfig()
    device = config.device

    train_loader, val_loader, test_loader = get_train_val_test_loaders(config)

    model = GPT2(config).to(device)

    optimizer = create_optimizer(model=model,
                                 learning_rate=config.learning_rate,
                                 weight_decay= config.weight_decay,
                                 betas= (config.beta1, config.beta2),
                                 eps= config.adam_eps)

    best_val_loss = float("inf")
    global_step = 0
    for epoch in range(1, config.epochs +1):
        if (
        config.max_steps is not None
        and global_step >= config.max_steps
    ):
            break
        train_loss, global_step = train_one_epoch(loader=train_loader,
                                     model=model,
                                     optimizer=optimizer,
                                     device=device,
                                     config=config,
                                     global_step=global_step,
                                     max_steps=config.max_steps)

        val_loss = validate(loader=val_loader,
                            model=model,
                            device=device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(path=config.checkpoint_path,
                            model=model,
                            optimizer=optimizer, 
                            epoch=epoch, 
                            global_step=global_step,
                            best_val_loss=best_val_loss,
                            config=config 
                            )

        print(
                f"Epoch [{epoch}/{config.epochs}] "
                f"Train Loss: {train_loss:.4f} "
                f"Val Loss: {val_loss:.4f} "
                f"Global Step: {global_step}"
            )

if __name__ == "__main__":
    main()


