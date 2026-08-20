from datasets import load_dataset

from dotenv import load_dotenv
import os
load_dotenv(override=True)
hf_token = os.getenv("HF_TOKEN")


ds = load_dataset("roneneldan/TinyStories", token=hf_token)

# above is the dataset for GPT1
#-------------------------------------------------------------------
from pathlib import Path

DATASET_NAME = "HuggingFaceFW/fineweb"
DATASET_CONFIG = "sample-10BT"

NUM_DOCUMENTS = 10_000

RAW_DATA_DIR = Path("data/raw")
OUTPUT_FILE = RAW_DATA_DIR / "fineweb_10k.txt"


# ---------------------------------------------------------
# Dataset Downloader
# ---------------------------------------------------------

def download_fineweb(
    num_documents: int = NUM_DOCUMENTS,
    output_file: Path = OUTPUT_FILE,
):
    """
    Stream a small subset of FineWeb and save raw documents locally.

    Args:
        num_documents: Number of documents to collect.
        output_file: Path where raw corpus will be saved.

    Returns:
        Number of documents saved.
    """

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Loading dataset: {DATASET_NAME}")
    print(f"Configuration: {DATASET_CONFIG}")
    print(f"Target documents: {num_documents}")

    dataset = load_dataset(
        DATASET_NAME,
        name=DATASET_CONFIG,
        split="train",
        streaming=True,
    )

    document_count = 0

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        for example in dataset:

            text = example.get("text", "")

            if not text:
                continue

            file.write(text.strip())
            file.write("\n\n")

            document_count += 1

            if document_count % 1000 == 0:
                print(
                    f"Downloaded {document_count:,} documents"
                )

            if document_count >= num_documents:
                break

    print("\nDataset download completed.")
    print(f"Documents saved : {document_count:,}")
    print(f"Output file     : {output_file}")

    return document_count


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    download_fineweb()