# GPT-1 From Scratch

A complete implementation of **GPT-1 (Generative Pre-trained Transformer)** from scratch using **PyTorch**, following the original OpenAI paper **"Improving Language Understanding by Generative Pre-Training" (2018)**.

This repository demonstrates the complete pipeline of building a decoder-only Transformer language model, including tokenizer training, data preprocessing, model implementation, training, inference, and evaluation.

> **Note:** The objective of this project is to understand and implement GPT-1 from first principles rather than reproduce the original model's performance.

---

# Project Highlights

* Implemented GPT-1 decoder-only Transformer from scratch.
* Built a custom **Byte Pair Encoding (BPE)** tokenizer.
* Created a complete data preprocessing and loading pipeline.
* Implemented Multi-Head Self-Attention with causal masking.
* Built decoder blocks with residual connections and Layer Normalization.
* Trained the model on the TinyStories dataset.
* Implemented multiple text generation strategies.
* Evaluated the model using quantitative and qualitative metrics.

---

# Repository Structure

```text
GPT-1/
│
├── artifacts/
│   └── tokenizer/
│
├── docs/
│   ├── 01_project_overview.md
│   ├── 02_dataset.md
│   ├── 03_tokenizer.md
│   ├── 04_data_pipeline.md
│   ├── 05_model.md
│   ├── 06_training.md
│   ├── 07_inference.md
│   └── 08_evaluation.md
│
├── scripts/
│   └── train_tokenizer.py
│
├── src/
│   ├── datasets/
│   ├── tokenizer/
│   ├── models/
│   ├── trainer/
│   ├── inference/
│   ├── evaluation/
│   ├── optim/
│   ├── losses/
│   └── utils/
│
├── tests/
│
├── train_pretrain.py
└── README.md
```

---

# Project Pipeline

```text
TinyStories Dataset
          │
          ▼
Text Cleaning
          │
          ▼
BPE Tokenizer Training
          │
          ▼
Tokenization
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
          │
          ▼
Training
          │
          ▼
Checkpoint Saving
          │
          ▼
Inference
          │
          ▼
Evaluation
```

---

# GPT-1 Architecture

The implemented model follows the original decoder-only Transformer architecture.

```text
Input Token IDs
        │
        ▼
Token Embedding
        │
        ▼
Positional Embedding
        │
        ▼
Decoder Block × N
        │
        ▼
Layer Normalization
        │
        ▼
Linear Projection
        │
        ▼
Vocabulary Logits
```

Each decoder block contains:

* Masked Multi-Head Self-Attention
* Feed Forward Network (MLP)
* Residual Connections
* Layer Normalization

---

# Dataset

The project uses the **TinyStories** dataset.

Due to hardware limitations, a subset of **10,000 stories** was used for both tokenizer training and GPT-1 training.

Dataset preprocessing includes:

* Text cleaning
* Tokenization using BPE
* Train/Validation split (80/20)
* Sliding window sequence generation

---

# Tokenizer

A custom **Byte Pair Encoding (BPE)** tokenizer was implemented from scratch.

Features:

* Vocabulary learning
* Merge rule generation
* Encoding
* Decoding
* Batch encoding/decoding
* Vocabulary serialization

---

# Training Configuration

| Parameter        |         Value |
| ---------------- | ------------: |
| Dataset          |   TinyStories |
| Stories Used     |        10,000 |
| Epochs           |             4 |
| Batch Size       |            16 |
| Learning Rate    |          3e-4 |
| Optimizer        |         AdamW |
| Loss Function    | Cross Entropy |
| Validation Split |           20% |

---

# Inference

The repository supports multiple decoding strategies:

* Greedy Sampling
* Temperature Sampling
* Top-k Sampling
* Top-p (Nucleus) Sampling

Generation is performed autoregressively by predicting one token at a time until either:

* `<EOS>` is generated, or
* the maximum sequence length is reached.

---

# Evaluation

The model was evaluated using both quantitative and qualitative approaches.

### Quantitative Evaluation

* Validation Loss
* Perplexity

### Qualitative Evaluation

Text generation using multiple prompts with different sampling strategies.

---

# Results

## Best Validation Loss

```text
≈ 5.4
```

## Perplexity

```text
258.99
```

## Observations

The model successfully learned:

* English sentence structure
* Story continuation
* Basic grammar
* Children's story patterns
* Next-token prediction

Because the model was trained on only **10,000 stories** for **4 epochs**, longer generations occasionally exhibit:

* Repetition
* Reduced coherence
* Limited vocabulary diversity
* BPE tokenization artifacts

These limitations are expected given the constrained training setup.

---

# How to Run

## 1. Clone the repository

```bash
git clone https://github.com/<your-username>/GPT-1-From-Scratch.git
cd GPT-1-From-Scratch
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Train the tokenizer

```bash
python scripts/train_tokenizer.py
```

## 4. Train GPT-1

```bash
python train_pretrain.py
```

## 5. Generate text

```bash
python src/inference/generate.py
```

## 6. Evaluate the model

```bash
python src/evaluation/evaluate.py
```

---

# Documentation

Detailed documentation is available in the `docs/` directory.

| Chapter | Description      |
| ------- | ---------------- |
| 01      | Project Overview |
| 02      | Dataset          |
| 03      | Tokenizer        |
| 04      | Data Pipeline    |
| 05      | GPT-1 Model      |
| 06      | Training         |
| 07      | Inference        |
| 08      | Evaluation       |

---

# Future Improvements

* Train on the complete TinyStories dataset.
* Increase the number of training epochs.
* Scale to GPT-2 architecture.
* Implement weight tying.
* Add KV Cache for faster inference.
* Implement Flash Attention.
* Add mixed precision (FP16/BF16) training.
* Support distributed and multi-GPU training.
* Fine-tune on downstream NLP tasks.

---

# References

1. Radford, A., et al. (2018). *Improving Language Understanding by Generative Pre-Training.*
2. Vaswani, A., et al. (2017). *Attention Is All You Need.*
3. TinyStories Dataset.

---

# License

This project is intended for educational and research purposes.