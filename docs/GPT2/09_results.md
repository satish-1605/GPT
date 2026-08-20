# GPT-2 — Results

## 1. Overview

This document records the results obtained from the GPT-2 implementation developed from scratch.

The goal was not to reproduce the exact performance of the original GPT-2 paper. Instead, the objective was to build and validate a complete GPT-2-style pretraining system:

```text
Tokenizer
   ↓
Dataset
   ↓
Data Pipeline
   ↓
GPT-2 Architecture
   ↓
Training
   ↓
Checkpointing
   ↓
Inference
   ↓
Evaluation
```

## 2. Model Configuration

The final model configuration used during development was approximately:

```text
Vocabulary Size     : 5,000
Context Length      : 128
Embedding Dimension : 256
Attention Heads     : 4
Transformer Layers  : 6
FFN Dimension       : 1,024
Dropout             : 0.1
```

Training configuration:

```text
Batch Size          : 16
Learning Rate       : 3e-4
Weight Decay        : 0.1
β₁                  : 0.9
β₂                  : 0.95
Warmup Steps        : 100
Minimum LR          : 3e-5
Gradient Clip Norm  : 1.0
```

## 3. Training Data

The GPT-2 phase moved from the smaller GPT-1 TinyStories setup toward a FineWeb-derived corpus.

The data pipeline included:

```text
Corpus
  ↓
Cleaning
  ↓
Train / Validation / Test Split
  ↓
GPT-2 Byte-Level BPE
  ↓
Token Streams
  ↓
Context Windows
  ↓
DataLoader
```

The pipeline was designed to be reusable for larger corpora.

## 4. Training Hardware

The GPT-2 implementation was trained locally.

This validated:

```text
Architecture
Data pipeline
Training loop
Optimizer
Scheduler
Gradient clipping
Checkpointing
Inference
Evaluation
```

Large-scale GPU infrastructure was not required to complete this phase.

## 5. Training Progress

The model was initially tested with smaller step counts before extending the training configuration to several thousand optimization steps.

The final recorded run reached:

```text
Global Step = 3,466
```

## 6. Final Training Metrics

The final training run reported:

```text
Train Loss : 5.6355
Val Loss   : 5.1117
Global Step: 3466
```

## 7. Final Evaluation Metrics

Independent evaluation produced:

```text
Validation Loss : 5.1459
Perplexity      : 171.7320
```

### Summary

| Metric          |   Result |
| --------------- | -------: |
| Global Steps    |    3,466 |
| Training Loss   |   5.6355 |
| Validation Loss |   5.1459 |
| Perplexity      | 171.7320 |

## 8. Inference

The trained model successfully performed autoregressive generation.

Supported decoding strategies:

```text
Greedy Sampling
Temperature Sampling
Top-k Sampling
Top-p Sampling
```

The model can generate text from a user-provided prompt.

## 9. What the Results Demonstrate

The results demonstrate that the complete GPT-2-style implementation is operational.

Specifically:

```text
✓ GPT-2 tokenizer
✓ Dataset preprocessing
✓ Train/Val/Test split
✓ Token streams
✓ Context windows
✓ Batch generation
✓ GPT-2 architecture
✓ Forward pass
✓ Next-token loss
✓ AdamW
✓ Learning-rate scheduling
✓ Gradient clipping
✓ Validation
✓ Checkpointing
✓ Model loading
✓ Autoregressive generation
✓ Multiple sampling strategies
✓ Perplexity evaluation
```

## 10. Comparison With GPT-1

The major achievement of the GPT-2 phase is the evolution from the earlier GPT-1-style implementation into a more complete GPT-2-style pretraining system.

```text
GPT-1
 ↓
Basic GPT training pipeline
 ↓
GPT-2
 ↓
Improved tokenizer
 ↓
Improved data pipeline
 ↓
Pre-LN architecture
 ↓
Final LayerNorm
 ↓
Residual initialization
 ↓
AdamW
 ↓
Learning-rate scheduling
 ↓
Gradient clipping
 ↓
Checkpoint/resume support
 ↓
Multiple decoding strategies
 ↓
Formal evaluation
```

## 11. Important Interpretation

The measured:

```text
Perplexity = 171.7320
```

should **not** be interpreted as the performance of the original GPT-2 model.

The original GPT-2 was trained with:

* Much larger model configurations
* Much more training data
* Much longer training
* Significantly greater compute

The current implementation is a small-scale reproduction designed for understanding and engineering validation.

## 12. Why the Model Is Not Yet GPT-2-Level in Capability

The architecture is GPT-2-style, but capability depends heavily on scale.

The current model has:

```text
d_model = 256
layers = 6
context = 128
vocab = 5,000
```

and was trained for:

```text
3,466 steps
```

Therefore, it should not be expected to demonstrate the broad knowledge and generalization capabilities of the original GPT-2.

Capability emerges from the combination of:

```text
Architecture
+
Parameters
+
Training Data
+
Training Tokens
+
Optimization
+
Compute
```

## 13. Purpose of This Result

The experiment establishes a working baseline before scaling.

```text
Phase 1
│
├── GPT-1              ✓
├── GPT-2 Improvements ✓
├── GPT-3 Scaling      → Next
└── Base GPT           → Later
```

## 14. Next Phase: GPT-3 Scaling

The GPT-3 phase will focus on:

```text
Model Size
Training Tokens
Batch Size
Training Steps
GPU Training
Compute Efficiency
Scaling Laws
```

This is where rented GPU infrastructure becomes significantly more useful.

The GPT-2 phase established the correctness of the system; the GPT-3 phase will investigate what happens when the same principles are scaled.

## 15. Final Result

The GPT-2 implementation successfully progressed from a small GPT-1-style model to a more complete GPT-2-style pretraining system.

Final recorded result:

```text
Steps           : 3,466
Train Loss      : 5.6355
Validation Loss : 5.1459
Perplexity      : 171.7320
```
