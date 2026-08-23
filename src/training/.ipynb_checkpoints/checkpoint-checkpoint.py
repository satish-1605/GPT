from pathlib import Path
import torch

def save_checkpoint(
        path:str|Path,
        model,
        optimizer,
        epoch :int,
        global_step :int,
        best_val_loss : float,
        config,
        ):
    checkpoint = {
        "model_state_dict":model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step":global_step,
        "best_val_loss": best_val_loss,
        "config": vars(config),
        }
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )
    torch.save(checkpoint, path)


def load_checkpoint(
        path:str|Path,
        model,
        optimizer,
        device
    ):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found : {path}"
        )

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(
        checkpoint['model_state_dict']
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    epoch = checkpoint["epoch"]
    global_step = checkpoint["global_step"]
    best_val_loss = checkpoint["best_val_loss"]

    return (
        epoch,
        global_step,
        best_val_loss,
    )


def load_model_checkpoint(
    path: str | Path,
    model,
    device,
):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model
