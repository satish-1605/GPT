from src.tokenizer.gpt2_bpe_trainer import GPT2BPETrainer

def test_bpe():
    trainer = GPT2BPETrainer(vocab_size=5000)

    sample_corpus = [
        "Hello world!",
        "GPT-2 is a language model."
    ]

    trainer.initialize_corpus(sample_corpus)
    trainer.initialize_vocabulary()

    print("Initial vocabulary:", len(trainer.vocabulary))

    trainer.train(verbose=True)

    print("\nFinal vocabulary size:", len(trainer.vocabulary))
    print("Number of merges:", len(trainer.merges))
    print("First merges:", trainer.merges[:10])