# Chapter 5: Model Architecture

## 1. Introduction

The model implemented in this project follows the **decoder-only Transformer architecture** introduced in the original GPT (Generative Pre-trained Transformer) paper. GPT-1 is an **autoregressive language model** that predicts the next token in a sequence by conditioning only on the previously observed tokens.

Unlike the original Transformer architecture proposed in *Attention Is All You Need*, GPT-1 uses **only the decoder stack** and employs **causal (masked) self-attention** to ensure that future tokens are never visible during training.

The overall architecture implemented in this project is illustrated below.

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
Add Embeddings
        │
        ▼
Decoder Block × N
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

The output logits represent the unnormalized probability scores for every token in the vocabulary. During inference, these logits are converted into probabilities to predict the next token.

---

# 2. Token Embeddings

Neural networks cannot directly process integer token IDs. Therefore, each token is first converted into a dense continuous vector known as a **token embedding**.

If the vocabulary size is **V** and the embedding dimension is **d_model**, then the embedding matrix has dimensions

```text
[V × d_model]
```

Each row of this matrix corresponds to the learned representation of a vocabulary token.

### Example

Input token IDs

```text
[15, 81, 22]
```

After the embedding lookup

```text
[
 e15
 e81
 e22
]
```

These dense vectors capture semantic and syntactic relationships between tokens and are learned during training through backpropagation.

---

# 3. Positional Embeddings

Unlike recurrent neural networks, Transformers process all tokens in parallel. As a result, they have no inherent notion of word order.

To incorporate positional information, GPT-1 uses **learnable positional embeddings**. A unique embedding vector is associated with every position in the sequence.

For the sentence

```text
The cat sat
```

the corresponding position IDs are

```text
0   1   2
```

Each position has its own learned embedding vector.

The final input representation is computed as

```text
Token Embedding
        +
Positional Embedding
```

This allows the model to distinguish between sentences such as

```text
Dog bites man
```

and

```text
Man bites dog
```

even though both contain the same words.

---

# 4. Decoder Blocks

The combined embeddings are passed through a stack of identical Transformer decoder blocks.

Each decoder block consists of the following components.

```text
Input
 │
 ▼
Masked Multi-Head Attention
 │
 ▼
Residual Connection
 │
 ▼
Layer Normalization
 │
 ▼
Feed Forward Network
 │
 ▼
Residual Connection
 │
 ▼
Layer Normalization
 │
 ▼
Output
```

If the model contains **N** decoder blocks, this process is repeated **N** times.

Each successive decoder block learns increasingly abstract contextual representations of the input sequence.

---

# 5. Masked Multi-Head Self-Attention

The self-attention mechanism enables each token to attend to previous tokens in the sequence and determine which tokens are most relevant for predicting the next token.

For every input token, three vectors are computed:

* Query (Q)
* Key (K)
* Value (V)

The attention mechanism computes the relevance of every previous token using these vectors.

For example,

```text
The animal didn't cross the street because it was tired.
```

The model learns that

```text
it
```

refers to

```text
animal
```

rather than

```text
street
```

by assigning higher attention weights to the appropriate token.

### Multi-Head Attention

Instead of computing a single attention distribution, GPT computes multiple attention heads in parallel.

Each head learns different linguistic relationships.

For example,

```text
Head 1 → Local syntax

Head 2 → Grammatical structure

Head 3 → Long-range dependencies

...

Head h
```

The outputs of all attention heads are concatenated and projected back to the model dimension.

This enables the model to capture multiple contextual relationships simultaneously.

---

# 6. Feed Forward Network (FFN)

After the attention layer, every token independently passes through a feed-forward neural network (FFN).

The FFN consists of two fully connected layers separated by a nonlinear activation function.

```text
Input
  │
  ▼
Linear
  │
  ▼
GELU
  │
  ▼
Linear
  │
  ▼
Output
```

Unlike the attention mechanism, the FFN processes each token independently while increasing the representational capacity of the model.

---

# 7. Residual Connections

Training deep neural networks becomes increasingly difficult due to vanishing gradients and optimization challenges.

GPT addresses this problem using **residual (skip) connections**.

Instead of learning

```text
F(x)
```

the model learns

```text
x + F(x)
```

where

* **x** is the input to a sublayer.
* **F(x)** is the transformation performed by that sublayer.

Residual connections improve gradient flow, stabilize optimization, and enable the successful training of deep Transformer architectures.

---

# 8. Layer Normalization

Layer Normalization improves training stability by normalizing the hidden representations across the embedding dimension.

Each decoder block contains two Layer Normalization operations:

* After the Multi-Head Self-Attention layer
* After the Feed Forward Network

An additional **final Layer Normalization** is applied after the last decoder block before the output projection.

The benefits of Layer Normalization include:

* Faster convergence
* Stable gradients
* Improved optimization
* Reduced training instability

---

# 9. Causal Mask

GPT-1 is an **autoregressive language model**, meaning that each token is predicted using only previously observed tokens.

For example,

```text
The cat sat on the _____
```

The model should only have access to

```text
The
cat
sat
on
the
```

It must **not** access

```text
mat
```

while predicting the missing token.

This constraint is enforced using a **causal attention mask**, which masks all future positions in the attention matrix.

An example causal mask is shown below.

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

Each token is allowed to attend only to itself and all previous tokens, ensuring correct next-token prediction during training.

---

# 10. Final Layer Normalization and Linear Projection

After passing through all decoder blocks, the final hidden representation is processed by:

1. A final Layer Normalization layer.
2. A Linear Projection layer.

The linear projection maps the hidden representation back to the vocabulary dimension.

For example,

```text
Hidden State (d_model)

        │
        ▼

Linear Projection

        │
        ▼

Vocabulary Size
```

The output is a vector of **logits**, where each value represents the score assigned to a vocabulary token.

During inference, these logits are converted into probabilities using the **Softmax** function. The next token is then selected using a decoding strategy such as:

* Greedy Decoding
* Temperature Sampling
* Top-k Sampling
* Top-p (Nucleus) Sampling

---

# 11. Summary

The GPT-1 model implemented in this project follows a **decoder-only Transformer architecture**. Input token IDs are first transformed into dense token embeddings and combined with learnable positional embeddings. These representations are processed by a stack of Transformer decoder blocks containing masked multi-head self-attention, feed-forward networks, residual connections, and layer normalization. After a final Layer Normalization layer, a linear projection produces vocabulary logits that are used to predict the next token in an autoregressive manner.

This architecture enables the model to learn contextual representations from large text corpora and generate coherent text one token at a time.
