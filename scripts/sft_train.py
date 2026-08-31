from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.alignment.sft.sft_config import SFTConfig
from src.alignment.sft.instruction_dataset import InstructionDataset
from src.tokenizer.hf_tokenizer import HFTokenizer
from src.models.gpt import GPT
from src.utils.config import GPTConfig
from src.alignment.sft.sft_trainer import SFTTrainer
 
def main():

    config = SFTConfig()

    device = torch.device(config.device)

    # Load tokenizer
    tokenizer = HFTokenizer(
        config.tokenizer_path
    )

    # Dataset
    train_dataset = InstructionDataset(
        Path(config.train_file),
        tokenizer,
        config.max_length
    )

    val_dataset = InstructionDataset(
        Path(config.val_file),
        tokenizer,
        config.max_length
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )

    # Model
    gpt_config = GPTConfig()

    model = GPT(gpt_config).to(device)

    # Load pretrained 300M checkpoint
    checkpoint = torch.load(
        config.base_checkpoint,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )

    # Smoke test
    trainer.train()


if __name__ == "__main__":
    main()
