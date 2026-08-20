from pathlib import Path
class DatasetPreprocessor:
    def __init__(self, min_length:int=50):
        self.min_length  = min_length


    def preprocess_document(self, document: str) -> str:
        """
        Prepare a single document while preserving
        its natural language content.
        """
        document = document.strip()
        return document

    def preprocess_corpus(
            self,
            input_file:str = Path,
            max_documents: int | None = None
        ) -> list[str]:

        """
        Load and preprocess the cleaned FineWeb corpus.

        Returns:
            List of cleaned documents.
        """
        input_file = Path(input_file)

        with input_file.open("r", encoding = "utf-8") as file:
            content = file.read()   

        documents = content.split("\n\n")

        processed_documents = []  

        for document in documents:
            document = self.preprocess_document(document)

            if len(document) < self.min_length:
                continue

            processed_documents.append(document)

            if (
                max_documents is not None
                and len(processed_documents) >= max_documents
            ):
                break

        return processed_documents







