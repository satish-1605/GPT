from src.datasets.preprocess import DatasetPreprocessor
from src.tokenizer.tokenizer import BPETokenizer
from src.datasets.download import ds
tokenizer = BPETokenizer.from_pretrained("artifacts/tokenizer")


preprocessor = DatasetPreprocessor(tokenizer)

processed = preprocessor.preprocess_corpus(
    ds["train"],
    max_stories=2
)

for ids in processed:
    print("==="*20)
    print(ids)
    print("==="*20)