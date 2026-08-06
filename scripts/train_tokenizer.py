from src.tokenizer.bpe import BPETrainer
from src.datasets.download import ds
from src.datasets.clean import clean_text
import time


NUM_STORIES = 10_000
VOCAB_SIZE = 5_000
SAVE_DIR = "artifacts/tokenizer"
def main():

    print("=" * 60)
    print("Preparing Training Corpus")
    print("=" * 60)

    start = time.time()

    corpus = [
        clean_text(ds["train"][i]["text"])
        for i in range(NUM_STORIES)
    ]

    print(f"Stories Loaded : {len(corpus):,}")
    print(f"Time Taken     : {time.time() - start:.2f} sec\n")

    trainer = BPETrainer(vocab_size=VOCAB_SIZE)

    trainer.fit(
        corpus,
        verbose=True,
        log_every=100,
    )

    trainer.save(SAVE_DIR)

    print("\nTokenizer saved successfully!")
    print(f"Location: {SAVE_DIR}")
    input("Press Enter to start training...")


if __name__ == "__main__":
    main()