# GPT-2 — Evaluation

## 1. Overview

Evaluation measures how well the trained GPT-2 model performs next-token prediction on unseen data.

The primary metrics are:

```text
Validation Loss
Perplexity
```

## 2. Evaluation Objective

For an input sequence:

```text
x₁ x₂ x₃ ... xₙ
```

the model predicts:

```text
x₂ x₃ x₄ ... xₙ₊₁
```

Predictions are compared against target tokens using cross-entropy loss.

## 3. Evaluation Dataset

The data pipeline provides:

```text
train_loader
val_loader
test_loader
```

The intended separation is:

```text
Training → parameter updates
Validation → model selection / monitoring
Test → final evaluation
```

## 4. Evaluation Mode

Before evaluation:

```python
model.eval()
```

Gradient calculation is disabled:

```python
with torch.no_grad():
```

Evaluation therefore does not modify model parameters.

## 5. Validation Loss

Validation loss is calculated over the validation DataLoader:

```text
Validation batches
       ↓
GPT-2
       ↓
Logits
       ↓
Cross Entropy
       ↓
Average Loss
```

The implementation calculates:

```text
Validation Loss
=
sum(batch losses) / number of batches
```

## 6. Perplexity

Perplexity is derived from language-model loss:

```text
Perplexity = exp(loss)
```

Lower perplexity indicates better predictive performance.

```text
Loss ↓
Perplexity ↓
```

## 7. Why Perplexity?

Perplexity provides an intuitive measure of the model's uncertainty about the next token.

A lower perplexity means the model assigns higher probability to the correct next tokens on average.

Perplexity should, however, be interpreted relative to:

* Dataset
* Tokenizer
* Vocabulary
* Context length
* Model size
* Evaluation protocol

## 8. Evaluation Script

The evaluation logic is exposed through:

```text
src/evaluation/evaluate.py
```

The script loads the trained model and evaluates it on validation data.

The output format is:

```text
==================================================
Validation Loss : ...
Perplexity      : ...
==================================================
```

## 9. Final Evaluation

The final local training run reached:

```text
Global Steps = 3,466
```

Independent evaluation produced:

```text
Validation Loss : 5.1459
Perplexity      : 171.7320
```

Training-time metrics were:

```text
Train Loss : 5.6355
Val Loss   : 5.1117
```

The difference between training-time validation loss and independently calculated validation loss is expected because they were obtained at different evaluation points or execution stages.

## 10. Qualitative Evaluation

Numerical metrics do not fully describe a generative language model.

Inference was therefore also tested using:

```text
Greedy
Temperature
Top-k
Top-p
```

This provides qualitative information about:

* Fluency
* Repetition
* Coherence
* Diversity
* Prompt continuation

## 11. Sampling Comparison

The same prompt can be passed through different sampling strategies.

For example:

```text
Prompt:
"Once upon a time there"
```

can be evaluated using:

```text
Greedy
Temperature
Top-k
Top-p
```

The goal is to understand how sampling changes generated output rather than declare one strategy universally superior.

## 12. Quantitative vs Qualitative Evaluation

The project uses two complementary approaches.

### Quantitative

```text
Validation Loss
Perplexity
```

### Qualitative

```text
Generated text
Sampling behavior
Coherence
Repetition
Diversity
```

## 13. GPT-2 Evaluation Scope

This phase focuses on pretraining evaluation rather than task-specific downstream benchmarks.

The model is evaluated on:

```text
Next-token prediction
```

rather than classification or question answering.

## 14. Why No Task-Specific Evaluation Yet?

The GPT-2 phase focuses on reproducing a pretrained autoregressive language-modeling setup.

The model is not instruction-tuned.

Therefore, task-specific evaluations such as:

```text
Question Answering
Summarization
Classification
Instruction Following
```

are outside this phase.

Those capabilities become more relevant in later phases involving instruction tuning and alignment methods.

## 15. Evaluation Limitations

### Dataset size

The training corpus is much smaller than the corpus used by the original GPT-2.

### Model size

The implemented model is significantly smaller than the original GPT-2 models.

### Training duration

The model was trained for only a few thousand optimization steps.

### Compute

Training was performed locally rather than using large-scale GPU infrastructure.

Therefore, these results demonstrate implementation and pipeline correctness rather than reproduction of original GPT-2 benchmark results.

## 16. Evaluation Takeaway

The evaluation confirms that the model performs the intended autoregressive language-modeling task.

Final metrics:

```text
Validation Loss = 5.1459
Perplexity      = 171.7320
```

These metrics provide a quantitative baseline for future scaling experiments.
