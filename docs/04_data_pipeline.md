# Chapter 4: Data Pipeline

## 1. Introduction

After training the tokenizer, the next step is to transform the raw text dataset into batches that can be efficiently consumed by the GPT-1 model during training. This entire process is referred to as the **data pipeline**.

The data pipeline converts raw stories into fixed-length token sequences, organizes them into training samples, and efficiently loads them into memory during model training and evaluation.

The complete pipeline implemented in this project is illustrated below.

```text
Raw Text (TinyStories)
          │
          ▼
   Text Preprocessing
          │
          ▼
     BPE Tokenizer
          │
          ▼
       Token IDs
          │
          ▼
 Train / Validation Split
          │
          ▼
 Sliding Window Dataset
          │
          ▼
      DataLoader
          │
          ▼
      GPT-1 Model
```

Each stage prepares the data for the next stage until it is ready to be fed into the GPT-1 model for next-token prediction.

---

# 2. Raw Text

The data pipeline begins with the **TinyStories** dataset. Each sample consists of a short English story stored as plain text.

### Example

```text
Once upon a time there was a little rabbit.
```

Due to hardware limitations, only the first **10,000 stories** were selected for this project.

These stories serve as the input for tokenizer training as well as GPT-1 model training.

---

# 3. Text Preprocessing

Before tokenization, each story undergoes a lightweight preprocessing step to ensure consistency across the dataset.

The preprocessing pipeline performs operations such as:

* Removing leading and trailing whitespace
* Normalizing line endings
* Removing empty samples
* Collapsing excessive blank lines

These steps produce clean and normalized text that is ready for tokenization.

---

# 4. BPE Tokenization

The cleaned stories are then passed through the custom **Byte Pair Encoding (BPE)** tokenizer developed in this project.

The tokenizer converts each story into a sequence of integer token IDs that can be processed by the GPT model.

### Example

**Input**

```text
Once upon a time
```

**Output**

```text
[421, 87, 1620, 53]
```

These token IDs become the actual input to the GPT-1 model during both training and inference.

---

# 5. Train / Validation Split

After tokenization, the processed stories are divided into training and validation sets using Scikit-learn's `train_test_split()`.

| Dataset    | Percentage |
| ---------- | ---------: |
| Training   |    **80%** |
| Validation |    **20%** |

The **training set** is used to optimize the model parameters, while the **validation set** is used to monitor the model's performance on unseen data and determine the best checkpoint during training.

---

# 6. Sliding Window Dataset

GPT models require input sequences of a fixed length. However, tokenized stories naturally have varying lengths.

To generate fixed-length training samples, the custom `GPTDataset` applies a **sliding window** over each tokenized story.

Assume the block size is **8**.

### Original Token Sequence

```text
[15, 42, 18, 90, 77, 61, 30, 11, 52, 83]
```

The first training sample becomes:

```text
Input : [15, 42, 18, 90, 77, 61, 30, 11]
Target: [42, 18, 90, 77, 61, 30, 11, 52]
```

The window then slides forward to generate additional training samples until the entire story has been processed.

This approach allows multiple training examples to be created from a single story while preserving the sequential structure of the text.

---

# 7. GPTDataset

The generated input-target pairs are stored in a custom PyTorch `GPTDataset`.

The dataset is responsible for:

* Storing tokenized stories
* Creating input-target pairs using the sliding window
* Returning one training sample at a time
* Supporting indexed access through `__getitem__()`
* Reporting the dataset size through `__len__()`

Each training example consists of an input sequence and its corresponding target sequence.

```text
Input Tokens
      │
      ▼
Target Tokens
(Shifted by One Position)
```

The target sequence is simply the input sequence shifted one token to the left, allowing the model to learn the **next-token prediction** objective.

---

# 8. DataLoader

The `GPTDataset` is finally wrapped inside a PyTorch `DataLoader`, which provides efficient batching and data loading during training.

The DataLoader performs several important tasks:

* Creates mini-batches
* Shuffles the training dataset
* Efficiently loads data during training
* Reduces data-loading overhead

The DataLoader configuration differs slightly for training and validation.

### Training DataLoader

```python
shuffle = True
```

The training samples are shuffled to improve generalization and reduce learning bias.

### Validation DataLoader

```python
shuffle = False
```

The validation dataset is not shuffled to ensure consistent evaluation across epochs.

Each iteration of the DataLoader returns a batch of input and target tensors that are fed directly into the GPT-1 model.

---

# 9. Complete Data Pipeline

The complete data flow implemented in this project can be summarized as follows.

```text
TinyStories Dataset
        │
        ▼
Text Cleaning
        │
        ▼
BPE Tokenizer
        │
        ▼
Token IDs
        │
        ▼
Train / Validation Split
        │
        ▼
GPTDataset
(Sliding Window)
        │
        ▼
PyTorch DataLoader
        │
        ▼
GPT-1 Model
```

---

# 10. Summary

The data pipeline transforms raw TinyStories text into batches suitable for GPT-1 training.

Raw stories are first cleaned and tokenized using a custom Byte Pair Encoding (BPE) tokenizer. The tokenized stories are then divided into training and validation sets. A sliding window is applied to generate fixed-length input-target pairs, which are stored in a custom `GPTDataset`. Finally, a PyTorch `DataLoader` creates mini-batches that are efficiently fed into the GPT-1 model during training and evaluation.

This modular pipeline provides a clean and reusable foundation for training decoder-only language models and can easily be extended to larger datasets and more advanced GPT architectures.