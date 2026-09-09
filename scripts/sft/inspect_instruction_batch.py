from pathlib import Path

from torch.utils.data import DataLoader
from src.alignment.sft.instruction_dataset import InstructionDataset
from src.tokenizer.hf_tokenizer import HFTokenizer

DATASET_PATH = Path("data/instruction/processed/train.jsonl")
TOKENIZER_PATH = Path("artifacts/tokenizer")

MAX_LENGTH = 512
BATCH_SIZE = 1

def main():
    tokenizer = HFTokenizer("artifacts/tokenizer/tokenizer.json")

    dataset = InstructionDataset(
        file_path=DATASET_PATH, 
        tokenizer=tokenizer, 
        max_length=MAX_LENGTH)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    batch = next(iter(dataloader))


    print(batch['labels'].shape)

    # print("=" * 60)
    # print("INSTRUCTION DATALOADER INSPECTION")
    # print("=" * 60)

    # print(
    #     f"Dataset size : {len(dataset)}"
    # )

    # print(
    #     f"Batch size   : "
    #     f"{batch['input_ids'].shape[0]}"
    # )

    # print(
    #     f"Sequence len : "
    #     f"{batch['input_ids'].shape[1]}"
    # )

    # print(
    #     f"input_ids shape : "
    #     f"{batch['input_ids'].shape}"
    # )

    # print(
    #     f"labels shape    : "
    #     f"{batch['labels'].shape}"
    # )

    # print(
    #     f"input_ids dtype : "
    #     f"{batch['input_ids'].dtype}"
    # )

    # print(
    #     f"labels dtype    : "
    #     f"{batch['labels'].dtype}"
    # )

    # print(
    #     f"Masked tokens in first example : "
    #     f"{(batch['labels'][0] == -100).sum().item()}"
    # )

    # print(
    #     f"Padding tokens in first example : "
    #     f"{
    #         (
    #             batch['input_ids'][0]
    #             == tokenizer.pad_token_id
    #         ).sum().item()
    #     }"
    # )

    # print("\nFirst 30 input IDs:")
    # print(
    #     batch["input_ids"][0][:30].tolist()
    # )

    # print("\nFirst 30 labels:")
    # print(
    #     batch["labels"][0][:30].tolist()
    # )

    # print("=" * 60)

if __name__ == "__main__":
    main()

