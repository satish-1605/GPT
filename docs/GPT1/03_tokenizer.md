# Chapter 3: Tokenizer

## 1. Introduction

Before a language model can process text, it must first convert raw text into a numerical representation. Neural networks cannot directly understand words or characters—they operate only on numerical values. A **tokenizer** bridges this gap by transforming text into a sequence of integer token IDs and converting those IDs back into human-readable text when required.

In this project, a **Byte Pair Encoding (BPE)** tokenizer was implemented from scratch and trained on a subset of **10,000 TinyStories**. The trained tokenizer was then used to preprocess the dataset for GPT-1 training and inference.

---

# 2. Why Tokenization?

Tokenization is one of the most fundamental steps in Natural Language Processing (NLP). It converts raw text into smaller units called **tokens**, which serve as the input to a language model.

Depending on the tokenization algorithm, a token may represent:

* A character
* A subword
* A complete word

Without tokenization, a language model has no mechanism to convert textual information into a numerical format that can be processed by neural networks.

The tokenizer serves several important purposes:

* Converts raw text into numerical token IDs.
* Reduces vocabulary size while preserving linguistic information.
* Handles unknown or rare words efficiently.
* Enables the model to process variable-length text.
* Reconstructs readable text from generated token IDs during inference.

### Example

**Input Text**

```text
The little cat sat on the mat.
```

**After Tokenization**

```text
[154, 982, 311, 478, 72, 145, 25]
```

These integer token IDs become the actual input to the GPT-1 model.

---

# 3. Byte Pair Encoding (BPE)

This project uses **Byte Pair Encoding (BPE)**, a subword tokenization algorithm widely adopted by modern language models, including GPT-2, RoBERTa, and many Hugging Face models.

Instead of treating every word as an independent token, BPE learns frequently occurring character sequences and gradually merges them into larger subword units.

This approach offers several advantages:

* Smaller vocabulary size
* Better handling of unseen or rare words
* Reduced number of unknown tokens
* Efficient representation of text
* Improved generalization across different words

### Example

Initially, the word

```text
lower
```

is represented as

```text
l o w e r
```

After several merge operations, it becomes

```text
low er
```

Eventually, the tokenizer learns the complete word as a single token:

```text
lower
```

As a result, frequently occurring words become single tokens, while infrequent words are represented as combinations of smaller subword tokens.

---

# 4. Vocabulary Creation

The tokenizer vocabulary was learned directly from the selected **10,000 TinyStories** used in this project.

The vocabulary creation process consists of the following steps:

1. Read all training stories.
2. Split each word into individual characters.
3. Count the frequency of adjacent token pairs.
4. Merge the most frequently occurring pair.
5. Repeat the merge process until the desired vocabulary size is reached.
6. Assign a unique integer ID to every token.

The final vocabulary contains:

* Individual characters
* Frequently occurring subwords
* Common words
* Special tokens required by the GPT model

Each token is assigned a unique integer ID, which is later used during GPT training and inference.

---

# 5. Merge Rules

The core idea behind BPE is the generation of **merge rules**.

During training, the tokenizer repeatedly identifies the most frequent adjacent pair of tokens and merges them into a new token.

### Example

Initial sequence:

```text
l o w e r
```

Frequent adjacent pairs:

```text
(l, o)
(o, w)
(w, e)
(e, r)
```

Suppose the pair

```text
(l, o)
```

appears most frequently.

The first merge produces

```text
lo w e r
```

Subsequent merges may produce

```text
low er
```

Eventually, the tokenizer learns

```text
lower
```

Each merge operation is stored as a **merge rule**. These learned rules are reused during encoding to ensure that new text is tokenized consistently.

---

# 6. Encoding

Encoding is the process of converting raw text into token IDs using the learned vocabulary and merge rules.

The encoding pipeline consists of the following steps:

1. Read the input text.
2. Split each word into characters.
3. Apply the learned BPE merge rules in order.
4. Replace each resulting token with its corresponding vocabulary ID.
5. Return the sequence of integer token IDs.

### Example

**Input**

```text
Once upon a time
```

**Encoded Output**

```text
[421, 87, 1620, 53]
```

These token IDs are passed directly to the GPT-1 model during both training and inference.

---

# 7. Decoding

Decoding performs the reverse operation of encoding.

It converts a sequence of token IDs back into readable text by mapping each ID to its corresponding token in the vocabulary and reconstructing the original sentence.

### Example

**Input IDs**

```text
[421, 87, 1620, 53]
```

**Decoded Output**

```text
Once upon a time
```

Decoding is an essential part of text generation because GPT models predict token IDs rather than words or characters.

---

# 8. Summary

A **Byte Pair Encoding (BPE)** tokenizer was implemented from scratch to preprocess the TinyStories dataset. The tokenizer was trained on **10,000 stories**, from which it learned both a vocabulary and a set of merge rules.

The tokenizer provides efficient **encoding** of raw text into token IDs and **decoding** of generated token IDs back into readable text. It serves as the first stage of the GPT-1 pipeline, enabling the language model to process textual data in a numerical format suitable for training and inference.