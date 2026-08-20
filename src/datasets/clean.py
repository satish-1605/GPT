import re

from pathlib import Path

RAW_DATA_FILE = Path("data/raw/fineweb_10k.txt")
CLEAN_DATA_FILE = Path("data/processed/fineweb_10k_clean.txt")

MIN_DOCUMENT_LENGTH = 50

def clean_document(text:str)-> str:
    """
    Clean a single FineWeb document.

    The cleaning is intentionally conservative because
    GPT-2 should learn from natural web text.
    """

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = "\n".join(
        line.rstrip()
        for line in text.split("\n")
    )

    text = re.sub(r"[ \t]+", " ", text)

    text = re.sub(r"\n{3,}", "\n\n", text)

    text = text.strip()

    return text

def clean_corpus(
        input_file :Path = RAW_DATA_FILE,
        output_file : Path = CLEAN_DATA_FILE,
        ):

    """
    Clean the raw FineWeb corpus and save the result.
    """

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    total_documents = 0
    kept_documents = 0
    removed_documents = 0

    with input_file.open("r", encoding="utf-8") as input_data:
        raw_documents = input_data.read().split("\n\n")

    with output_file.open("w", encoding="utf-8") as output_data:
        for document in raw_documents:
            total_documents += 1
            document = clean_document(document)

            if not document:
                removed_documents += 1
                continue

            if len(document) < MIN_DOCUMENT_LENGTH:
                removed_documents += 1
                continue

            output_data.write(document)
            output_data.write("\n\n")

            kept_documents += 1

    print("\nCleaning completed.")
    print(f"Total documents   : {total_documents:,}")
    print(f"Kept documents    : {kept_documents:,}")
    print(f"Removed documents : {removed_documents:,}")
    print(f"Output file       : {output_file}")

    return {
        "total": total_documents,
        "kept": kept_documents,
        "removed": removed_documents,
    }


if __name__ == "__main__":

    clean_corpus()

    




