# GPT-2 — Overview

## 1. Introduction

This phase extends the GPT-1 implementation into a more GPT-2-style autoregressive language model.

The objective is not to reproduce the original GPT-2 training run at its original scale, but to implement and understand the major architectural, tokenization, data-pipeline, training, inference, and evaluation improvements introduced in the GPT-2 generation of the GPT family.

The implementation is built from scratch using Python and PyTorch, following software-engineering principles such as modular design, reusable components, unit testing, checkpointing, and separation of concerns.

The overall objective is:

```text
GPT-1
   ↓
GPT-2 Improvements
   ↓
GPT-3 Scaling
   ↓
Base GPT
```

---

## 2. GPT-2 Objective

GPT-2 remains an autoregressive language model.

Given a sequence of tokens:

```text
x₁, x₂, x₃, ..., xₜ
```

the model learns to predict the next token:

```text
x₂, x₃, x₄, ..., xₜ₊₁
```

The training objective is the standard causal language-modeling loss:

```text
L = -Σ log P(xₜ | x₁, ..., xₜ₋₁)
```

The model is therefore trained using **next-token prediction**, rather than a task-specific objective.

The intention is that sufficiently large-scale pretraining on diverse text allows the model to learn general language patterns that can later support zero-shot or few-shot behavior.

---

## 3. What Changes from GPT-1?

The GPT-2 implementation improves several parts of the GPT-1 system.

### GPT-1

```text
Dataset
   ↓
Tokenizer
   ↓
Tokenized samples
   ↓
Transformer
   ↓
Training
   ↓
Generation
```

### GPT-2

```text
FineWeb / WebText-like corpus
          ↓
Document preprocessing
          ↓
GPT-2 style byte-level BPE
          ↓
EOS-aware token streams
          ↓
Context windows
          ↓
GPT-2 Transformer
          ↓
AdamW + LR scheduling
          ↓
Checkpointing + validation
          ↓
Inference + decoding
          ↓
Loss + perplexity
```

The major improvements implemented in this phase are:

* Byte-level encoding
* GPT-2 style pre-tokenization
* BPE tokenizer training
* Explicit end-of-text handling
* Improved document and corpus processing
* Sliding/context window construction
* Input/target alignment
* GPT-2 style Pre-LN Transformer blocks
* Final LayerNorm
* Residual initialization
* AdamW optimizer
* Learning-rate warmup and decay
* Gradient clipping
* Step-based training
* Checkpointing and checkpoint loading
* Validation
* Multiple decoding strategies
* Perplexity evaluation

---

## 4. GPT-2 Tokenizer

The tokenizer is implemented as a byte-level BPE tokenizer.

The pipeline is:

```text
Raw text
   ↓
Byte encoding
   ↓
GPT-2 pre-tokenization
   ↓
BPE merges
   ↓
Token IDs
```

The tokenizer implementation contains components for:

* Byte encoding
* GPT-2 pre-tokenization
* Vocabulary management
* BPE merge learning
* Token encoding
* Token decoding
* Tokenizer artifact persistence

The tokenizer also handles the special end-of-text token:

```text
<|endoftext|>
```

This allows document boundaries to be represented explicitly in the tokenized corpus.

---

## 5. Dataset

Instead of the small TinyStories-based setup used during GPT-1 development, the GPT-2 implementation uses a **FineWeb-derived WebText-like corpus**.

The dataset pipeline performs:

```text
Raw corpus
   ↓
Cleaning
   ↓
Document preprocessing
   ↓
Train / Validation / Test split
   ↓
Tokenization
   ↓
Token streams
```

The corpus is divided into training, validation, and test sets before model training.

This provides separate data for:

* Parameter optimization
* Validation during development
* Final held-out evaluation

---

## 6. Context Window Construction

GPT-2 training uses fixed-length context windows.

For a context length of `128`, an example can be constructed as:

```text
Input:
t₀ t₁ t₂ ... t₁₂₇

Target:
t₁ t₂ t₃ ... t₁₂₈
```

Therefore, every input token corresponds to the next token that the model must predict.

The implementation separates:

* Window index construction
* Input/target alignment
* Dataset representation
* DataLoader creation

This makes the data pipeline independently testable.

---

## 7. GPT-2 Model Architecture

The GPT-2 model is implemented as a causal Transformer.

The major architectural changes from the GPT-1 implementation include:

### Pre-LayerNorm

Normalization is applied before the attention and feed-forward sublayers.

Conceptually:

```text
x
 ↓
LayerNorm
 ↓
Self-Attention
 ↓
Residual Add
 ↓
LayerNorm
 ↓
Feed Forward
 ↓
Residual Add
```

### Final LayerNorm

A final LayerNorm is applied after the Transformer blocks before projecting the hidden representation to vocabulary logits.

### Residual Initialization

Residual projections use the GPT-2-style initialization scaling intended to control the magnitude of residual activations as depth increases.

### Causal Self-Attention

The model can only attend to previous tokens and the current token.

```text
Token 1 → Token 1
Token 2 → Token 1, Token 2
Token 3 → Token 1, Token 2, Token 3
...
```

Future tokens are masked during training.

---

## 8. Training

The GPT-2 training pipeline uses:

* AdamW
* Weight decay
* Learning-rate warmup
* Learning-rate decay
* Gradient clipping
* Global step tracking
* Validation
* Checkpointing

The training loop is step-aware rather than relying only on epochs.

A training step corresponds to **one optimizer update using one batch**.

For example, with:

```text
batch_size = 16
```

one step processes one batch containing 16 sequences and performs one:

```text
forward
   ↓
loss
   ↓
backward
   ↓
gradient clipping
   ↓
optimizer update
```

The global step is incremented after each optimizer update.

---

## 9. Checkpointing

Training checkpoints store the state required to preserve the training process.

The checkpoint contains:

```text
Model state
Optimizer state
Epoch
Global step
Best validation loss
Configuration
```

This allows the model to be:

* Loaded for inference
* Evaluated later
* Used for continued training
* Resumed without losing optimizer state

The best validation checkpoint is selected based on validation loss.

---

## 10. Inference

The inference pipeline separates text generation from sampling strategies.

The generation process is:

```text
Prompt
   ↓
Tokenizer
   ↓
Token IDs
   ↓
GPT-2
   ↓
Next-token logits
   ↓
Sampling strategy
   ↓
Next token
   ↓
Append token
   ↓
Repeat
```

The implementation supports multiple decoding approaches:

### Greedy Decoding

Selects the highest-probability token.

### Temperature Sampling

Adjusts the probability distribution before sampling.

### Top-K Sampling

Restricts sampling to the `K` highest-probability tokens.

### Top-P Sampling

Restricts sampling to the smallest set of tokens whose cumulative probability exceeds `P`.

These strategies allow the effect of decoding choices on generated text to be studied independently of the trained model.

---

## 11. Evaluation

The primary evaluation objective remains **next-token prediction**.

The model is evaluated using:

### Loss

Cross-entropy loss over the held-out data.

### Perplexity

Perplexity is calculated as:

```text
Perplexity = exp(loss)
```

A lower perplexity indicates that the model assigns higher probability to the observed test tokens.

Generation is also evaluated qualitatively using the different decoding strategies.

The evaluation therefore combines:

```text
Quantitative
    ↓
Loss
Perplexity

Qualitative
    ↓
Generated text
Decoding behavior
```

---

## 12. Experimental Results

The small-scale implementation was trained on the available FineWeb-derived corpus.

One of the completed training runs reached:

```text
Global Step : 3466
Train Loss  : 5.6355
Validation Loss : 5.1459
Perplexity  : 171.7320
```

The model showed a clear reduction in validation loss as training progressed.

For example, the observed validation perplexity improved substantially between earlier checkpoints and the later 3,466-step checkpoint.

Generation experiments also demonstrated that the model learned basic language and token-level patterns, although generation quality remained limited by the relatively small model size, dataset size, and training compute.

---

## 13. Limitations

This implementation should not be interpreted as a reproduction of the original GPT-2 training run.

The original GPT-2 work used substantially larger:

* Dataset
* Model
* Training compute
* Number of training tokens
* Training duration

The implementation in this repository is intended primarily to reproduce and understand the **engineering and modeling principles** behind GPT-2 at a scale that can be developed and tested locally.

Therefore, poor or repetitive generation at this scale does not necessarily indicate an architectural failure.

The experiments demonstrate the progression from:

```text
Architecture
   ↓
Training
   ↓
Learning
   ↓
Generation
```

rather than attempting to reproduce the original GPT-2 capability level.

---

## 14. GPT-1 → GPT-2 Summary

| Component         | GPT-1                 | GPT-2 Implementation                 |           |    |
| ----------------- | --------------------- | ------------------------------------ | --------- | -- |
| Objective         | Next-token prediction | Next-token prediction                |           |    |
| Tokenization      | BPE-based             | Byte-level BPE                       |           |    |
| Pre-tokenization  | Basic                 | GPT-2 style                          |           |    |
| Special token     | EOS handling          | Explicit `<                          | endoftext | >` |
| Dataset           | TinyStories           | FineWeb-derived corpus               |           |    |
| Data construction | Fixed samples         | Context windows + stride             |           |    |
| Transformer       | GPT-style             | GPT-2-style improvements             |           |    |
| Normalization     | GPT-1 implementation  | Pre-LN + final LayerNorm             |           |    |
| Initialization    | Standard              | Residual-aware initialization        |           |    |
| Optimizer         | AdamW                 | AdamW + weight decay                 |           |    |
| LR                | Fixed/configured      | Warmup + decay                       |           |    |
| Gradients         | Basic                 | Gradient clipping                    |           |    |
| Checkpointing     | Basic                 | Model + optimizer + training state   |           |    |
| Inference         | Generation            | Greedy + Temperature + Top-K + Top-P |           |    |
| Evaluation        | Loss                  | Loss + Perplexity                    |           |    |

---

## 15. Position in the Overall GPT Roadmap

The GPT-2 implementation is the second major stage of the GPT-from-scratch project.

```text
Phase 1
│
├── GPT-1                  ✅
├── GPT-2 Improvements     ✅
├── GPT-3 Scaling          ⏳
└── Base GPT               ⏳
```

The next major stage is **GPT-3 Scaling**, where the focus shifts from architectural improvements toward understanding the relationship between:

```text
Model Size
     +
Dataset Size
     +
Training Tokens
     +
Compute
     ↓
Model Capability
```

This is also where GPU-based training becomes substantially more useful and allows scaling experiments that are impractical on the current CPU setup.

---

## 16. Key Takeaway

The GPT-2 phase demonstrates that a decoder-only Transformer can be extended from the GPT-1 implementation into a more robust autoregressive language-modeling system by improving:

```text
Tokenizer
    +
Dataset
    +
Context construction
    +
Transformer architecture
    +
Training infrastructure
    +
Inference
    +
Evaluation
```

