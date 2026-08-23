# GPT From Scratch

A complete implementation of **GPT-1 and GPT-2-style decoder-only Transformer language models from scratch using PyTorch**.

This repository follows the core ideas introduced in OpenAI's original GPT work and progressively builds the system from tokenizer and data preprocessing through model architecture, training, inference, and evaluation.

> **Note:** The objective of this project is to understand and implement GPT architectures from first principles rather than reproduce the original GPT-1 or GPT-2 model performance.

---

## Project Highlights

* Implemented GPT-1 decoder-only Transformer from scratch.
* Extended the implementation toward a GPT-2-style architecture.
* Built a custom **Byte Pair Encoding (BPE)** tokenizer.
* Created reusable dataset and data-loading pipelines.
* Implemented Multi-Head Self-Attention with causal masking.
* Implemented Transformer decoder blocks with residual connections and Layer Normalization.
* Added AdamW optimization, learning-rate scheduling, and gradient clipping.
* Implemented checkpoint saving and resume support.
* Added multiple autoregressive text-generation strategies.
* Evaluated models using validation loss, perplexity, and qualitative generation.
* Designed the codebase to provide a foundation for future GPT-3-style scaling experiments.

---

# Project Evolution

The project is developed progressively, moving from a small GPT-1 implementation toward GPT-2-style improvements.

```text
GPT-1
 │
 ├── Custom BPE Tokenizer
 ├── TinyStories Dataset
 ├── Decoder-only Transformer
 ├── Causal Self-Attention
 ├── Training Pipeline
 └── Text Generation
        │
        ▼
GPT-2
 │
 ├── Improved Data Pipeline
 ├── GPT-2-style Architecture
 ├── FineWeb-derived Data
 ├── AdamW
 ├── Learning-Rate Scheduling
 ├── Gradient Clipping
 ├── Checkpoint / Resume
 ├── Multiple Sampling Strategies
 └── Formal Evaluation
        │
        ▼
Future GPT-3 Scaling
```

The GPT-1 phase establishes the fundamental architecture and training pipeline. The GPT-2 phase then improves the data, architecture, optimization, and evaluation components while keeping the implementation understandable and practical for local experimentation.

---

# Project Pipeline

The overall workflow is:

```text
Dataset
   │
   ▼
Text Cleaning
   │
   ▼
BPE Tokenizer
   │
   ▼
Tokenization
   │
   ▼
Train / Validation / Test Split
   │
   ▼
Sliding-Window / Context Dataset
   │
   ▼
DataLoader
   │
   ▼
GPT Model
   │
   ▼
Training
   │
   ▼
Checkpoint
   │
   ▼
Inference
   │
   ▼
Evaluation
```

---

# GPT Architecture

Both implementations use the decoder-only Transformer concept.

```text
Input Token IDs
       │
       ▼
Token Embedding
       │
       ▼
Positional Information
       │
       ▼
Transformer Decoder Blocks
       │
       ▼
Final Layer Normalization
       │
       ▼
Linear Projection
       │
       ▼
Vocabulary Logits
```

Each Transformer block contains:

* Causal Multi-Head Self-Attention
* Feed-Forward Network / MLP
* Residual Connections
* Layer Normalization

The GPT-2 phase additionally incorporates GPT-2-style architectural and optimization improvements.

---

# GPT-1

## Dataset

The GPT-1 implementation uses the **TinyStories** dataset.

Due to local hardware constraints, a subset of:

```text
10,000 stories
```

was used for tokenizer training and model training.

The preprocessing pipeline includes:

* Text cleaning
* BPE tokenization
* 80/20 train-validation split
* Sliding-window sequence generation
* DataLoader construction

## Training Configuration

| Parameter        |         Value |
| ---------------- | ------------: |
| Dataset          |   TinyStories |
| Stories Used     |        10,000 |
| Epochs           |             4 |
| Batch Size       |            16 |
| Learning Rate    |          3e-4 |
| Optimizer        |         AdamW |
| Loss             | Cross Entropy |
| Validation Split |           20% |

## GPT-1 Results

The best recorded validation metrics were approximately:

```text
Validation Loss ≈ 5.4
Perplexity       = 258.99
```

The model successfully learned basic:

* English sentence structure
* Story continuation
* Grammar patterns
* Children's story patterns
* Next-token prediction

Because the model was trained on only 10,000 stories for four epochs, longer generations can exhibit repetition, reduced coherence, limited vocabulary diversity, and BPE artifacts.

---

# GPT-2

The GPT-2 phase extends the original implementation toward a more complete GPT-2-style pretraining system.

## Dataset

The GPT-2 phase moved from the smaller TinyStories experiment toward a **FineWeb-derived corpus**.

The data pipeline follows:

```text
Corpus
  ↓
Cleaning
  ↓
Train / Validation / Test Split
  ↓
BPE Tokenization
  ↓
Token Streams
  ↓
Context Windows
  ↓
DataLoader
```

The pipeline is designed to be reusable for larger training corpora.

## Model Configuration

The local GPT-2-style experiment used approximately:

| Parameter           | Value |
| ------------------- | ----: |
| Vocabulary Size     | 5,000 |
| Context Length      |   128 |
| Embedding Dimension |   256 |
| Attention Heads     |     4 |
| Transformer Layers  |     6 |
| FFN Dimension       | 1,024 |
| Dropout             |   0.1 |
| Batch Size          |    16 |

## Optimization

The GPT-2 training pipeline uses:

| Parameter          | Value |
| ------------------ | ----: |
| Optimizer          | AdamW |
| Learning Rate      |  3e-4 |
| Weight Decay       |   0.1 |
| β₁                 |   0.9 |
| β₂                 |  0.95 |
| Warmup Steps       |   100 |
| Minimum LR         |  3e-5 |
| Gradient Clip Norm |   1.0 |

The training system also supports:

* Validation during training
* Best-checkpoint selection
* Global-step tracking
* Checkpoint resume
* Learning-rate scheduling
* Gradient clipping

## GPT-2 Training Result

The final recorded local training run reached:

```text
Global Steps = 3,466
Train Loss   = 5.6355
Val Loss     = 5.1117
```

Independent evaluation produced:

```text
Validation Loss = 5.1459
Perplexity      = 171.7320
```

These results are intended as an engineering and implementation baseline rather than a reproduction of the original GPT-2 benchmark performance.

The model is considerably smaller and was trained on far less data and compute than the original GPT-2 models.

---

# Inference

Both GPT implementations support autoregressive text generation.

The generation process is:

```text
Prompt
  ↓
Tokenizer
  ↓
Token IDs
  ↓
GPT Model
  ↓
Logits
  ↓
Sampling
  ↓
Next Token
  ↓
Append Token
  ↓
Repeat
  ↓
Decode
  ↓
Generated Text
```

Supported decoding strategies include:

* **Greedy Sampling**
* **Temperature Sampling**
* **Top-k Sampling**
* **Top-p / Nucleus Sampling**

Generation stops when an end-of-text token is produced or the configured generation limit is reached.

---

# Evaluation

The project uses both quantitative and qualitative evaluation.

## Quantitative Evaluation

The primary metrics are:

* Validation Loss
* Perplexity

Perplexity is calculated as:

```text
Perplexity = exp(loss)
```

Lower validation loss and perplexity indicate better next-token prediction on the evaluation data.

## Qualitative Evaluation

Generated text is evaluated using different prompts and sampling strategies to examine:

* Fluency
* Coherence
* Repetition
* Diversity
* Prompt continuation
* Sampling behavior

---

# Results Summary

| Model | Dataset         | Steps / Epochs | Validation Loss | Perplexity |
| ----- | --------------- | -------------: | --------------: | ---------: |
| GPT-1 | TinyStories     |       4 epochs |           ≈ 5.4 |     258.99 |
| GPT-2 | FineWeb-derived |    3,466 steps |          5.1459 |   171.7320 |

These results should be interpreted within the context of each experiment's dataset size, model size, tokenizer, training duration, and available compute.

They are not directly comparable to the original GPT-1 or GPT-2 benchmark results.

---

# How to Run

## 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/GPT-From-Scratch.git
cd GPT-From-Scratch
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Train the Tokenizer

```bash
python scripts/train_tokenizer.py
```

## 4. Train the Model

```bash
python train_pretrain.py
```

## 5. Generate Text

```bash
python src/inference/generate.py
```

## 6. Evaluate the Model

```bash
python src/evaluation/evaluate.py
```

---

# Documentation

Detailed implementation notes are available in the `docs/` directory.

| Chapter | Description        |
| ------- | ------------------ |
| 01      | Project Overview   |
| 02      | Dataset            |
| 03      | Tokenizer          |
| 04      | Data Pipeline      |
| 05      | Model Architecture |
| 06      | Training           |
| 07      | Inference          |
| 08      | Evaluation         |
| 09      | Results            |

---

# Future Improvements

The next stages of the project will focus on scaling and improving the existing implementation.

* Train on larger and more diverse corpora.
* Scale the model toward GPT-3-style configurations.
* Increase context length.
* Investigate scaling laws.
* Implement weight tying where appropriate.
* Add KV caching for faster inference.
* Implement Flash Attention.
* Add mixed-precision FP16/BF16 training.
* Support distributed and multi-GPU training.
* Improve checkpoint management.
* Add downstream NLP evaluation.
* Experiment with instruction tuning and alignment.

---

# Learning Goals

This project is intended to provide a practical understanding of how modern autoregressive language models are constructed.

The implementation covers the complete lifecycle:

```text
Tokenizer
   ↓
Data
   ↓
Transformer
   ↓
Optimization
   ↓
Training
   ↓
Checkpointing
   ↓
Inference
   ↓
Evaluation
```

The progression from GPT-1 to GPT-2 is also intended to demonstrate that improvements in language-model capability are not determined by architecture alone. Model size, training data, optimization, training tokens, context length, and compute all contribute to the final result.

---

# References

1. Radford, A., et al. (2018). *Improving Language Understanding by Generative Pre-Training.*
2. Radford, A., et al. (2019). *Language Models are Unsupervised Multitask Learners.*
3. Vaswani, A., et al. (2017). *Attention Is All You Need.*
4. TinyStories Dataset.
5. FineWeb Dataset.

---

# License

This project is intended for educational and research purposes.
