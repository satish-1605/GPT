from src.datasets.clean import clean_text
from src.datasets.download import ds

from src.tokenizer.tokenizer import BPETokenizer

class DatasetPreprocessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


    def preprocess_story(self, story: str) -> list[int]:
        story = clean_text(story)
        ids = self.tokenizer.encode(story)
        bos_id = self.tokenizer.token_to_id["<BOS>"]
        eos_id = self.tokenizer.token_to_id["<EOS>"]
        return [bos_id] + ids + [eos_id]

    def preprocess_corpus(
            self,
            dataset,
            max_stories: int | None = None
        ) -> list[list[int]]:
        processed_stories = []
        if max_stories is None:
            max_stories = len(dataset)

        for i in range(max_stories):
            story = dataset[i]["text"]
            token_ids = self.preprocess_story(story)
            processed_stories.append(token_ids)
        return processed_stories






