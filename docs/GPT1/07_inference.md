# Chapter 7: Inference

## 1. Introduction

After training, the GPT-1 model is used for **inference**, also known as **text generation**. During inference, the model receives an input prompt and predicts the next token in the sequence. The newly generated token is appended to the input, and the updated sequence is fed back into the model to predict the following token.

This autoregressive process continues until either the maximum sequence length is reached or an end-of-sequence (**`<EOS>`**) token is generated.

The complete inference pipeline implemented in this project is illustrated below.

```text id="z8t2am"
Input Prompt
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
GPT-1 Model
      │
      ▼
Vocabulary Logits
      │
      ▼
Sampling Strategy
      │
      ▼
Next Token
      │
      ▼
Append to Input
      │
      ▼
Repeat
      │
      ▼
Decode Tokens
      │
      ▼
Generated Text
```

The GPT-1 model always produces **logits**, which represent unnormalized scores for every token in the vocabulary. These logits are converted into probabilities using the **Softmax** function.

The probability of selecting token *i* is given by

[
P(x_i)=\frac{e^{z_i}}{\sum_{j=1}^{V}e^{z_j}}
]

where

* (z_i) is the logit corresponding to token *i*.
* (V) is the vocabulary size.

The sampling strategy determines how the next token is selected from this probability distribution.

---

# 2. Autoregressive Text Generation

GPT-1 generates text **one token at a time**.

At every generation step:

1. The current prompt is tokenized.
2. The token IDs are passed through the GPT-1 model.
3. The model predicts logits for every vocabulary token.
4. A sampling strategy selects the next token.
5. The predicted token is appended to the prompt.
6. The updated prompt becomes the input for the next iteration.

This process repeats until one of the following stopping conditions is met:

* The maximum sequence length is reached.
* The `<EOS>` token is generated.

The generation loop implemented in this project is illustrated below.

```text id="mcv8zd"
Prompt
   │
   ▼
Tokenizer
   │
   ▼
GPT-1
   │
   ▼
Predict Next Token
   │
   ▼
Append Token
   │
   ▼
Repeat
```

---

# 3. Greedy Sampling

Greedy Sampling is the simplest decoding strategy.

At every generation step, the token with the **highest probability** is selected.

Mathematically,

[
x=\arg\max_i P(x_i)
]

### Example

| Token  | Probability |
| ------ | ----------: |
| cat    |        0.62 |
| dog    |        0.20 |
| rabbit |        0.10 |
| bird   |        0.08 |

The generated token is

```text id="lx4p2l"
cat
```

### Advantages

* Fast and computationally efficient
* Deterministic
* Produces consistent outputs

### Disadvantages

* Can generate repetitive text
* Limited diversity
* May become trapped in repetitive loops

---

# 4. Temperature Sampling

Temperature Sampling adjusts the randomness of the probability distribution before sampling.

The logits are scaled as

[
z_i'=\frac{z_i}{T}
]

where

* (T) is the temperature.
* (T=1) leaves the distribution unchanged.

The scaled logits are then passed through the Softmax function.

## Low Temperature (T < 1)

Example

```text id="9jskzd"
Temperature = 0.3
```

Probability distribution

```text id="vzn4v3"
cat     0.90
dog     0.06
rabbit  0.03
bird    0.01
```

A lower temperature produces more deterministic outputs.

---

## High Temperature (T > 1)

Example

```text id="7c3nww"
Temperature = 1.5
```

Probability distribution

```text id="focj0o"
cat     0.40
dog     0.25
rabbit  0.20
bird    0.15
```

A higher temperature produces more diverse and creative text.

### Advantages

* Adjustable randomness
* Easy to implement
* Commonly used for language generation

### Disadvantages

* High temperatures may generate incoherent text.
* Very low temperatures behave similarly to greedy decoding.

---

# 5. Top-k Sampling

Top-k Sampling restricts the candidate tokens to the **k most probable** tokens.

All remaining tokens are discarded.

### Example

Original probabilities

| Token  | Probability |
| ------ | ----------: |
| cat    |        0.45 |
| dog    |        0.25 |
| rabbit |        0.15 |
| bird   |        0.10 |
| fish   |        0.05 |

If

```text id="nggvj5"
k = 3
```

the remaining candidate tokens become

```text id="o7m0ij"
cat
dog
rabbit
```

Their probabilities are normalized before one token is sampled.

### Advantages

* Prevents extremely unlikely words
* Produces more diverse outputs than greedy decoding
* Frequently used in practical language generation

### Disadvantages

* Requires selecting an appropriate value of *k*.
* A fixed value of *k* may exclude useful candidate tokens.

---

# 6. Top-p (Nucleus) Sampling

Top-p (Nucleus) Sampling dynamically selects the **smallest set of tokens whose cumulative probability exceeds a threshold (p)**.

Unlike Top-k Sampling, the number of candidate tokens varies according to the probability distribution.

### Example

| Token  | Probability | Cumulative |
| ------ | ----------: | ---------: |
| cat    |        0.45 |       0.45 |
| dog    |        0.25 |       0.70 |
| rabbit |        0.15 |       0.85 |
| bird   |        0.10 |       0.95 |
| fish   |        0.05 |       1.00 |

If

```text id="18n34i"
p = 0.90
```

the selected candidate set becomes

```text id="h4jg1s"
cat
dog
rabbit
bird
```

These probabilities are normalized before one token is sampled.

### Advantages

* Dynamically adapts to the probability distribution
* Produces fluent and diverse text
* Avoids selecting extremely unlikely tokens
* Widely adopted in modern LLMs

### Disadvantages

* Slightly more computationally expensive than greedy decoding
* Requires selecting an appropriate value of *p*

---

# 7. Comparison of Sampling Strategies

| Method      | Deterministic | Randomness |  Creativity | Typical Use                |
| ----------- | :-----------: | :--------: | :---------: | -------------------------- |
| Greedy      |      Yes      |    None    |     Low     | Deterministic generation   |
| Temperature |       No      | Adjustable | Medium–High | Controlling randomness     |
| Top-k       |       No      |  Moderate  |     High    | General-purpose generation |
| Top-p       |       No      |  Adaptive  |     High    | Modern LLM inference       |

---

# 8. Inference Workflow

The complete inference process implemented in this project is summarized below.

```text id="3gwpbs"
Input Prompt
      │
      ▼
Clean Text
      │
      ▼
BPE Tokenizer
      │
      ▼
Token IDs
      │
      ▼
GPT-1 Forward Pass
      │
      ▼
Vocabulary Logits
      │
      ▼
Sampling Strategy
      │
      ▼
Next Token
      │
      ▼
Append to Prompt
      │
      ▼
Repeat Until <EOS> or Maximum Length
      │
      ▼
Decode Tokens
      │
      ▼
Generated Text
```

---

# 9. Summary

During inference, the trained GPT-1 model generates text in an **autoregressive** manner by predicting one token at a time. Each generated token is appended to the existing prompt and fed back into the model until an end-of-sequence token or the maximum sequence length is reached.

The model first converts the input prompt into token IDs using the custom BPE tokenizer, produces vocabulary logits through a forward pass, and then applies a sampling strategy to select the next token.

This project implements four decoding strategies:

* **Greedy Sampling**
* **Temperature Sampling**
* **Top-k Sampling**
* **Top-p (Nucleus) Sampling**

These strategies provide different trade-offs between determinism, diversity, and creativity, allowing the behavior of the language model to be adapted for different text generation tasks.