# Chapter 1: Project Overview

## 1. Introduction

**GPT-1 (Generative Pre-trained Transformer 1)** was the first model in the GPT family, introduced by OpenAI in 2018 in the research paper **"Improving Language Understanding by Generative Pre-Training."** It demonstrated that a single language model could learn general language understanding by first training on large amounts of unlabeled text and then adapting to downstream tasks through fine-tuning.

GPT-1 is built upon the **Transformer decoder architecture** proposed in the paper **"Attention Is All You Need."** Unlike traditional machine learning approaches that require training a separate model for each task, GPT-1 learns a generic representation of language that can be transferred to multiple Natural Language Processing (NLP) tasks.

---

## 2. GPT-1 Training Strategy

GPT-1 follows a **two-stage training process**.

### Stage 1: Pre-training

During pre-training, the model learns the statistical structure of language by predicting the next token in a sequence using a large corpus of unlabeled text.

**Example**

```text
Input  : The cat sat on the
Target : mat
```

Through this self-supervised objective, the model learns grammar, syntax, semantics, reasoning patterns, and long-range dependencies without requiring manually labeled data.

---

### Stage 2: Fine-tuning

After pre-training, the learned language representations are adapted to specific downstream tasks using comparatively smaller labeled datasets.

Common downstream tasks include:

* Sentiment Analysis
* Text Classification
* Question Answering
* Natural Language Inference
* Semantic Similarity

This transfer learning approach significantly reduces the amount of labeled data required for individual NLP tasks.

---

## 3. Training Objective

The primary objective of GPT-1 during pre-training is **next-token prediction**.

Given a sequence of previous tokens,

[
w_1, w_2, \ldots, w_{t-1}
]

the model predicts the probability of the next token,

[
P(w_t \mid w_1, w_2, \ldots, w_{t-1})
]

where:

* (w_1, w_2, ..., w_{t-1}) represent the previously observed tokens.
* (w_t) is the next token that the model attempts to predict.

The training objective can be visualized as:

```text
                Large Unlabeled Text
                        │
                        ▼
                 Pre-training
            (Next Token Prediction)
                        │
                        ▼
               Pre-trained GPT-1
                        │
                        ▼
                 Fine-tuning
             (Task-Specific Data)
                        │
                        ▼
                 Downstream Tasks
        • Text Classification
        • Question Answering
        • Natural Language Inference
        • Semantic Similarity
```

---

# 4. Objective of This Repository

The objective of this repository is to implement the **GPT-1 architecture from scratch** while understanding every major component involved in building a decoder-only language model.

The primary focus of this project is **learning and implementation**, rather than reproducing the exact performance of the original GPT-1 model.

Throughout this repository, the complete pipeline has been implemented, including:

* Training a custom Byte Pair Encoding (BPE) tokenizer.
* Preparing the TinyStories dataset for language modeling.
* Building the GPT-1 decoder architecture entirely from scratch using PyTorch.
* Training the model using next-token prediction.
* Saving and loading model checkpoints.
* Implementing multiple decoding strategies for text generation.
* Performing both quantitative and qualitative evaluation of the trained model.

The repository demonstrates the complete lifecycle of a GPT-style language model—from raw text preprocessing to text generation.

---

# 5. Project Structure

The repository is organized into modular components, each responsible for a specific stage of the GPT-1 pipeline.

## artifacts/

Stores all generated artifacts required during training and inference.

```text
artifacts/
└── tokenizer/
    ├── vocab.json
    ├── merges.txt
    └── config.json
```

These files contain:

* Vocabulary
* BPE merge rules
* Tokenizer configuration

---

## scripts/

### `train_tokenizer.py`

Responsible for training the Byte Pair Encoding (BPE) tokenizer on the selected TinyStories corpus.

---

## src/datasets/

This module implements the complete data preparation pipeline.

### `download.py`

Downloads the TinyStories dataset from Hugging Face.

### `clean.py`

Cleans and normalizes the raw text before tokenization.

### `preprocess.py`

Preprocesses individual stories and the complete corpus into tokenized sequences.

### `dataset.py`

Creates training samples using the sliding-window approach and provides indexed access through a custom PyTorch Dataset.

### `dataloader.py`

Creates PyTorch DataLoaders for efficient mini-batch training.

### `data_pipeline.py`

Combines all preprocessing steps into a reusable pipeline by:

* Initializing the tokenizer
* Preprocessing the corpus
* Splitting the data into training and validation sets
* Creating Dataset objects
* Returning the corresponding DataLoaders

---

## src/tokenizer/

### `bpe.py`

Implements the complete Byte Pair Encoding training algorithm and handles saving/loading of vocabulary and merge rules.

### `tokenizer.py`

Provides utilities for:

* Encoding text into token IDs
* Decoding token IDs back into text
* Batch encoding
* Batch decoding

---

## src/models/

Contains the complete GPT-1 architecture.

### `embedding.py`

Implements token embeddings and learnable positional embeddings.

### `attention.py`

Implements masked multi-head self-attention used for contextual representation learning.

### `mlp.py`

Implements the feed-forward neural network used inside each Transformer block.

### `layer_norm.py`

Implements Layer Normalization for stable optimization and faster convergence.

### `block.py`

Combines attention, feed-forward network, residual connections, dropout, and layer normalization into a single Transformer decoder block.

### `gpt.py`

Assembles the complete GPT-1 model using multiple decoder blocks.

---

## Training

### `train_pretrain.py`

Responsible for:

* Model training
* Validation
* Loss computation
* Optimizer updates
* Checkpoint saving
* Monitoring training progress

---

## src/inference/

Implements text generation using the trained GPT model.

Supported decoding strategies include:

* Greedy Decoding
* Temperature Sampling
* Top-k Sampling
* Top-p (Nucleus) Sampling

---

## src/evaluation/

Implements both quantitative and qualitative evaluation.

### Quantitative Evaluation

* Validation Loss
* Perplexity

### Qualitative Evaluation

Generates text for a fixed set of prompts using multiple decoding strategies for comparison.

---

## optim/

Contains the implementation and configuration of the **AdamW optimizer**.

---

## losses/

Contains utilities for computing the language modeling loss during training and validation.

---

## src/utils/

Provides reusable helper utilities, including:

* Configuration management
* Checkpoint saving and loading
* Causal mask generation
* Miscellaneous utility functions

---

## tests/

Contains unit tests for validating the correctness of individual components implemented throughout the repository.

---

## Summary

This repository demonstrates the complete implementation of a decoder-only GPT-1 language model from scratch. Every major component—including tokenization, data preprocessing, Transformer architecture, training pipeline, inference, and evaluation—has been implemented manually to provide a comprehensive understanding of how modern autoregressive language models work internally.