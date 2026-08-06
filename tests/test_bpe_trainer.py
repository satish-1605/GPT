# from src.datasets.download import ds
from src.datasets.clean import clean_text
from src.datasets.bpe import BPETrainer


corpus = [
        "low lower lowest",
        "new newer newest"
    ]

trainer = BPETrainer(vocab_size=20)

assert trainer.vocab_size == 20
assert trainer.corpus == []
assert trainer.vocabulary == set()
assert trainer.merges == []
assert len(trainer.pair_counts) == 0

# ##  Testing Corpus Initialization
trainer.initialize_corpus(corpus)

assert trainer.corpus[0] == ['l', 'o', 'w']
assert trainer.corpus[1] == ['l', 'o', 'w', 'e', 'r']
assert trainer.corpus[2] == ['l', 'o', 'w', 'e', 's', 't']

assert trainer.corpus[3] == ['n', 'e', 'w']
assert trainer.corpus[4] == ['n', 'e', 'w', 'e', 'r']

# # ##  Testing vocab Initialization
trainer.initialize_vocabulary()

expected = {
    'l',
    'o',
    'w',
    'e',
    'r',
    's',
    't',
    'n'
}

assert trainer.vocabulary == expected

# # ## Testing pair counting
pairs = trainer.count_pair_frequencies()

assert trainer.pair_counts[('l', 'o')] == 3
assert trainer.pair_counts[('o', 'w')] == 3

assert trainer.pair_counts[('n', 'e')] == 3
assert trainer.pair_counts[('e', 'w')] == 3

pair = trainer.get_best_pair()

assert pair is not None
assert pair in [
    ('l', 'o'),
    ('o', 'w'),
    ('n', 'e'),
    ('w', 'e')
]

# # ## merging the best pair and updating the corpus
print(trainer.merge_pair(pairs))




