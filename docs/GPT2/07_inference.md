# GPT-2 — Inference

## 1. Overview

The GPT-2 inference pipeline generates text autoregressively.

The model receives a prompt and repeatedly predicts the next token.

```text
Prompt
  ↓
Tokenizer
  ↓
Token IDs
  ↓
GPT-2
  ↓
Logits
  ↓
Sampling
  ↓
Next Token
  ↓
Append Token
  ↓
GPT-2
  ↓
...
```

The inference implementation is located in:

```text
src/inference/
├── load_checkpoint.py
├── sampling.py
└── generate.py
```

## 2. Loading the Model

Inference loads:

1. GPT-2 configuration
2. Trained tokenizer
3. GPT-2 model
4. Trained model checkpoint

The model is switched to evaluation mode:

```python
model.eval()
```

Gradient computation is disabled using:

```python
torch.no_grad()
```

## 3. Prompt Processing

The input prompt is cleaned and tokenized:

```text
Raw Prompt
    ↓
Cleaning
    ↓
GPT-2 Tokenizer
    ↓
Token IDs
```

For example:

```text
"Once upon a time there"
```

becomes a sequence of integer token IDs.

## 4. Autoregressive Generation

For each generation step:

```text
Current sequence
      ↓
Take last context_length tokens
      ↓
GPT-2 forward pass
      ↓
Last-position logits
      ↓
Sampling strategy
      ↓
Next token
      ↓
Append
```

The generated token becomes part of the input for the next prediction.

## 5. Context Window

The model uses:

```python
context_length = 128
```

If the generated sequence becomes longer than the context window, only the most recent tokens are passed to the model:

```python
context_ids = generated_ids[-context_length:]
```

This keeps the model input within the supported sequence length.

## 6. Sampling Strategies

The inference implementation supports:

```text
Greedy
Temperature
Top-k
Top-p
```

These are implemented in:

```text
src/inference/sampling.py
```

## 7. Greedy Sampling

Greedy decoding selects the token with the highest probability.

```text
logits
   ↓
argmax
   ↓
highest-scoring token
```

Advantages:

* Deterministic
* Simple
* Easy to debug

Disadvantages:

* Can produce repetitive text
* Can become overly predictable

## 8. Temperature Sampling

Temperature modifies the sharpness of the probability distribution:

```text
logits / T
```

Lower temperature:

```text
T < 1
```

produces a sharper distribution.

Higher temperature:

```text
T > 1
```

produces greater diversity.

## 9. Top-k Sampling

Top-k sampling restricts candidates to the `k` most probable tokens.

For:

```python
k = 10
```

the model samples only from the ten highest-probability tokens.

```text
Vocabulary
    ↓
Select top 10
    ↓
Normalize probabilities
    ↓
Sample
```

## 10. Top-p Sampling

Top-p, or nucleus sampling, selects the smallest set of tokens whose cumulative probability exceeds `p`.

For example:

```python
p = 0.8
```

selects candidates accounting for approximately 80% of the probability mass.

Unlike top-k, the number of candidates can change at every generation step.

## 11. End-of-Text Handling

Generation checks for:

```text
<|endoftext|>
```

If this token is generated, generation stops.

```text
Generate token
     ↓
Is EOS?
 ┌───┴───┐
Yes     No
 ↓       ↓
Stop   Continue
```

## 12. Generation Length

The current generation loop uses the configured context length as the maximum number of generation iterations.

For a production-quality interface, a dedicated parameter such as:

```python
max_new_tokens
```

can independently control generation length.

## 13. Inference Examples

Greedy:

```python
generate(
    "Once upon a time there",
    sampling_strategy="greedy_sampling",
)
```

Temperature:

```python
generate(
    prompt,
    sampling_strategy="temp_sampling",
    T=0.8,
)
```

Top-k:

```python
generate(
    prompt,
    sampling_strategy="top_k_sampling",
    k=10,
)
```

Top-p:

```python
generate(
    prompt,
    sampling_strategy="top_p_sampling",
    p=0.8,
)
```

## 14. GPT-2 Inference Pipeline

```text
Prompt
  ↓
Clean
  ↓
Encode
  ↓
Context Window
  ↓
GPT-2
  ↓
Logits
  ↓
Sampling
  ↓
Token ID
  ↓
Append
  ↓
Repeat
  ↓
Decode
  ↓
Generated Text
```

## 15. Important Limitation

The current model is a **small-scale GPT-2-style implementation**, not the original GPT-2 model.

It was trained on a relatively small corpus and for a limited number of optimization steps. Inference quality should therefore not be compared directly with the original GPT-2 model.

The purpose of this implementation is to reproduce and understand the architecture and training methodology at a scale practical for local development.
