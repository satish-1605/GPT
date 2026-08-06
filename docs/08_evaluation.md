## 1. Introduction

Evaluation is an essential step in assessing the performance of a language model. Unlike traditional machine learning tasks, where predictions are compared against fixed labels, language models are evaluated using both numerical metrics and the quality of the generated text.

In this project, the trained GPT-1 model is evaluated using two complementary approaches:

1. **Quantitative Evaluation**

   * Validation Loss
   * Perplexity

2. **Qualitative Evaluation**

   * Text generation using multiple prompts
   * Comparison of different sampling strategies

The quantitative evaluation is performed on the validation dataset, while the qualitative evaluation is performed by generating text from user-defined prompts.

---

# 2. Evaluation Pipeline

The evaluation workflow implemented in this project is shown below.

```text id="u2dr3v"
Load Best Checkpoint
        │
        ▼
Validation Dataset
        │
        ▼
Forward Pass
        │
        ▼
Validation Loss
        │
        ▼
Perplexity
        │
        ▼
Load Prompt
        │
        ▼
Text Generation
        │
        ▼
Sampling Strategy
        │
        ▼
Generated Response
```

The evaluation pipeline first measures the model quantitatively using the validation dataset and then evaluates the quality of generated text using multiple decoding strategies.

---

# 3. Quantitative Evaluation

Quantitative evaluation measures the model's prediction performance using numerical metrics.

The metrics implemented in this project are:

* Validation Loss
* Perplexity

---

## 3.1 Validation Loss

Validation loss measures how accurately the model predicts the next token on unseen validation data.

During evaluation:

* The model is switched to evaluation mode using `model.eval()`.
* Gradient computation is disabled using `torch.no_grad()`.
* The validation dataset is passed through the trained model.
* The average Cross Entropy Loss is computed across all validation batches.

A lower validation loss indicates better generalization.

The validation loss is computed as

[
Loss=-\frac{1}{N}\sum_{i=1}^{N}\log(P(y_i))
]

where

* (N) is the total number of predicted tokens.
* (y_i) is the correct target token.
* (P(y_i)) is the predicted probability assigned to that token.

### Result

| Metric               |                                       Value |
| -------------------- | ------------------------------------------: |
| Best Validation Loss | **≈ 5.4** *(Replace with your exact value)* |
| Best Epoch           |              **4** *(Replace if different)* |

The best checkpoint used for inference is selected based on the lowest validation loss.

---

## 3.2 Perplexity

Perplexity is one of the most widely used evaluation metrics for autoregressive language models.

It measures how confidently the model predicts the next token and is computed directly from the validation loss.

[
Perplexity=e^{Loss}
]

Lower perplexity values indicate better predictive performance.

### Result

| Metric          |                                                       Value |
| --------------- | ----------------------------------------------------------: |
| Validation Loss | **≈ 5.56** *(Replace with exact value used for evaluation)* |
| Perplexity      |                                                 **258.996** |

Although the perplexity is relatively high compared to production-scale language models, it is expected because the model was trained on only **10,000 TinyStories** using limited computational resources.

---

# 4. Qualitative Evaluation

Numerical metrics alone cannot fully describe the quality of generated text.

Therefore, the model is also evaluated qualitatively by generating text from multiple prompts using different sampling strategies.

The generated outputs are analyzed based on:

* Grammar
* Sentence coherence
* Story consistency
* Vocabulary usage
* Repetition
* Creativity

---

# 5. Evaluation Prompts

The following prompts are used in this project.

```text id="g7u3d9"
Once upon a time
There was a little
One day
The little girl
The little boy
The cat was
The dog ran
A friendly rabbit
The king lived
In the forest
```

Each prompt is evaluated using all four decoding strategies.

---

# 6. Sampling Strategy Comparison

For every prompt, the following decoding strategies are evaluated:

* Greedy Sampling
* Temperature Sampling
* Top-k Sampling
* Top-p (Nucleus) Sampling

This allows the generated responses to be compared under different inference settings.

---

# 7. Observations

The qualitative evaluation produced the following observations.

### Greedy Sampling

* Produces deterministic outputs.
* Generates grammatically stable sentences.
* Frequently repeats common words and phrases.
* Less creative than probabilistic decoding methods.

---

### Temperature Sampling

* Introduces controlled randomness.
* Produces more diverse outputs.
* Higher temperatures may reduce coherence.

---

### Top-k Sampling

* Generates more diverse text while avoiding highly improbable words.
* Produces a good balance between creativity and consistency.
* Suitable for general-purpose text generation.

---

### Top-p (Nucleus) Sampling

* Dynamically adapts to the model's probability distribution.
* Produces the most natural and diverse outputs.
* Occasionally generates less coherent text when the probability distribution is highly uncertain.

---

# 8. Comparison of Sampling Strategies

| Method      | Diversity | Coherence | Repetition | Typical Behavior         |
| ----------- | :-------: | :-------: | :--------: | ------------------------ |
| Greedy      |    Low    |    High   |    High    | Deterministic generation |
| Temperature |   Medium  |   Medium  |   Medium   | Adjustable randomness    |
| Top-k       |    High   |    High   |   Medium   | Balanced generation      |
| Top-p       | Very High |   Medium  |     Low    | Diverse generation       |

---

# 9. Discussion

The evaluation demonstrates that the GPT-1 implementation successfully learned the general structure of the TinyStories dataset.

The model is capable of generating:

* Grammatically structured sentences
* Basic story progression
* Character interactions
* Simple narrative sequences

Several limitations were also observed:

* Repetition during long generations
* Reduced coherence over extended text
* Limited vocabulary diversity
* Difficulty maintaining long-range context

These limitations are expected because the model was trained on only **10,000 stories** for a small number of epochs due to hardware constraints.

Large language models are typically trained on billions of tokens using significantly larger model architectures and computational resources.

---

# 10. Summary

The GPT-1 implementation was evaluated using both quantitative and qualitative methods.

Quantitatively, the model was evaluated using **Validation Loss** and **Perplexity**, providing numerical measures of prediction performance.

Qualitatively, multiple prompts were evaluated using **Greedy**, **Temperature**, **Top-k**, and **Top-p** sampling strategies to compare the generated text.

Although the model was trained on a relatively small subset of the TinyStories dataset, the evaluation demonstrates that the complete GPT-1 pipeline—from training and inference to text generation—was successfully implemented and is capable of generating coherent story-like text using multiple decoding strategies.
