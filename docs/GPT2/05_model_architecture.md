# GPT-2 — Model Architecture

## 1. Overview

The GPT-2 implementation follows the decoder-only Transformer architecture introduced by GPT-2, with several architectural improvements over the GPT-1 implementation.

The model is an **autoregressive language model**.

Its objective is:

```text
P(x₁, x₂, ..., xₙ)
```

which is factorized as:

```text
P(x₁) × P(x₂ | x₁) × P(x₃ | x₁, x₂) × ... × P(xₙ | x₁, ..., xₙ₋₁)
```

During training, the model receives a sequence of tokens and predicts the next token at every position.

The high-level architecture is:

```text
Token IDs
    ↓
Token Embedding
    +
Position Embedding
    ↓
Dropout
    ↓
GPT-2 Transformer Blocks × N
    ↓
Final LayerNorm
    ↓
Language Modeling Head
    ↓
Logits
```

---

# 2. GPT-2 Architecture vs GPT-1

The GPT-2 implementation retains the core decoder-only Transformer architecture from GPT-1 but introduces important architectural changes.

| Component               | GPT-1                    | GPT-2                          |
| ----------------------- | ------------------------ | ------------------------------ |
| Architecture            | Decoder-only Transformer | Decoder-only Transformer       |
| Normalization           | Post-LN                  | Pre-LN                         |
| Final LayerNorm         | Not used                 | Used                           |
| Residual initialization | Standard                 | Scaled residual initialization |
| Causal attention        | Yes                      | Yes                            |
| Self-attention          | Multi-head               | Multi-head                     |
| Feed-forward network    | Yes                      | Yes                            |
| Positional encoding     | Learned                  | Learned                        |
| Weight tying            | Implementation dependent | GPT-style LM head              |
| Depth                   | Configurable             | Configurable                   |
| Vocabulary              | GPT-1 tokenizer          | GPT-2 byte-level BPE           |

The most important architectural improvement implemented here is **Pre-LayerNorm**.

---

# 3. Model Configuration

The architecture is controlled through `GPTConfig`.

The current configuration includes:

```python
vocab_size = 5000
max_seq_len = 128

d_model = 256
num_heads = 4
num_layers = 6

ff_hidden_dim = 1024

dropout = 0.1
eps = 1e-5
bias = True
```

These values define the size and structure of the model.

---

# 4. Vocabulary Size

The model receives integer token IDs produced by the tokenizer.

With:

```python
vocab_size = 5000
```

the valid token IDs belong to:

```text
0 ... 4999
```

The embedding layer therefore has approximately:

```text
vocab_size × d_model
```

parameters.

For the current configuration:

```text
5000 × 256
= 1,280,000
```

embedding parameters.

---

# 5. Token Embeddings

The first transformation performed by the model is token embedding.

Input:

```text
[batch_size, sequence_length]
```

Example:

```text
[16, 128]
```

The embedding layer converts every token ID into a dense vector.

With:

```text
d_model = 256
```

the resulting representation is:

```text
[16, 128, 256]
```

Conceptually:

```text
Token IDs
    ↓
Embedding Lookup
    ↓
Token Representations
```

---

# 6. Positional Embeddings

Self-attention does not inherently know the order of tokens.

Therefore, positional information is added to the token representations.

The GPT-2 implementation uses **learned positional embeddings**.

For:

```text
max_seq_len = 128
d_model = 256
```

the positional embedding table has:

```text
128 × 256
```

parameters.

The input representation becomes:

```text
Token Embedding
       +
Position Embedding
       ↓
Combined Representation
```

---

# 7. Input Representation

After token and positional embeddings are combined:

```text
x = token_embedding + position_embedding
```

The shape is:

```text
[batch_size, sequence_length, d_model]
```

For the current configuration:

```text
[16, 128, 256]
```

Dropout is then applied before entering the Transformer blocks.

---

# 8. Transformer Block

The GPT-2 model consists of multiple identical Transformer blocks.

Current configuration:

```python
num_layers = 6
```

Therefore:

```text
Input
  ↓
Block 1
  ↓
Block 2
  ↓
Block 3
  ↓
Block 4
  ↓
Block 5
  ↓
Block 6
  ↓
Final LayerNorm
```

Each block contains:

```text
Pre-LayerNorm
     ↓
Causal Multi-Head Self-Attention
     ↓
Residual Connection
     ↓
Pre-LayerNorm
     ↓
Feed-Forward Network
     ↓
Residual Connection
```

---

# 9. Pre-LayerNorm

One of the major changes from the GPT-1 implementation is the placement of LayerNorm.

The GPT-2 block follows the Pre-LN structure:

```text
x
│
├───────────────┐
│               │
↓               │
LayerNorm       │
↓               │
Self-Attention  │
↓               │
└────── + x ────┘
        │
        ↓
        x
```

Then:

```text
x
│
├───────────────┐
│               │
↓               │
LayerNorm       │
↓               │
Feed Forward    │
↓               │
└────── + x ────┘
        │
        ↓
       output
```

This differs from the Post-LN structure used in the GPT-1 implementation.

---

# 10. Why Pre-LN?

Pre-LN provides a more stable optimization path for deeper Transformer networks.

The residual path can pass information through the network without requiring the normalization layer to be applied after the residual addition.

Conceptually:

```text
Post-LN:

x → Attention → + Residual → LayerNorm
```

while:

```text
Pre-LN:

x → LayerNorm → Attention → + Residual
```

The GPT-2 implementation uses the second arrangement.

This architectural decision becomes increasingly important when the model is scaled to greater depth.

---

# 11. LayerNorm

LayerNorm is applied before both major sublayers:

```text
Attention
Feed Forward
```

The configuration uses:

```python
eps = 1e-5
```

The epsilon value prevents numerical instability when computing normalization.

Conceptually:

```text
LayerNorm(x)
```

normalizes the hidden representation across its feature dimension.

---

# 12. Causal Self-Attention

The attention mechanism allows each token to interact with previous tokens.

For autoregressive language modeling, a token must not access future tokens.

Therefore the model uses a **causal attention mask**.

For a sequence:

```text
A B C D
```

the allowed attention pattern is:

```text
A → A
B → A B
C → A B C
D → A B C D
```

The forbidden relationships are:

```text
A → B C D
B → C D
C → D
```

This guarantees that prediction at position `t` only uses information available up to position `t`.

---

# 13. Query, Key and Value

For an input representation:

```text
X
```

the attention mechanism creates:

```text
Q = XWq
K = XWk
V = XWv
```

The attention scores are:

```text
QKᵀ
```

scaled by the dimensionality of the attention head:

```text
QKᵀ / √d_head
```

The causal mask is then applied before softmax.

The resulting attention weights are multiplied by `V`.

Conceptually:

```text
Q, K, V
   ↓
QKᵀ / √d_head
   ↓
Causal Mask
   ↓
Softmax
   ↓
Attention Weights
   ↓
Weighted V
```

---

# 14. Multi-Head Attention

Instead of performing attention once, the model divides the representation into multiple attention heads.

Current configuration:

```python
num_heads = 4
d_model = 256
```

Therefore:

```text
d_head = d_model / num_heads
       = 256 / 4
       = 64
```

Each head operates on a 64-dimensional representation.

The heads are then concatenated:

```text
Head 1
Head 2
Head 3
Head 4
   ↓
Concatenate
   ↓
Output Projection
```

The final attention output returns to:

```text
[batch_size, sequence_length, d_model]
```

or:

```text
[16, 128, 256]
```

---

# 15. Attention Residual Connection

After self-attention:

```text
attention_output
```

is added to the original block input:

```text
x = x + attention_output
```

This is the first residual connection.

The resulting representation is then passed to the feed-forward sublayer.

---

# 16. Feed-Forward Network

The feed-forward network operates independently on each token position.

The architecture is:

```text
d_model
   ↓
Linear
   ↓
Activation
   ↓
Linear
   ↓
d_model
```

Current configuration:

```python
d_model = 256
ff_hidden_dim = 1024
```

Therefore:

```text
256 → 1024 → 256
```

The feed-forward network expands the representation by a factor of:

```text
1024 / 256 = 4
```

This is the standard Transformer-style expansion used in this implementation.

---

# 17. Feed-Forward Activation

The feed-forward network uses the configured activation function implemented by the GPT-2 architecture.

The purpose of the nonlinear activation is to allow the network to learn nonlinear transformations of the token representations.

Conceptually:

```text
Input
  ↓
Linear
  ↓
Nonlinearity
  ↓
Linear
  ↓
Output
```

---

# 18. Feed-Forward Residual Connection

The feed-forward output is added back to the residual stream:

```text
x = x + feed_forward_output
```

The complete block therefore becomes:

```text
x
│
├──────────────────────────────┐
│                              │
↓                              │
LayerNorm                       │
↓                              │
Causal Self-Attention           │
↓                              │
└────────────── + x ────────────┘
               │
               ↓
               x
               │
               ├──────────────────────┐
               │                      │
               ↓                      │
           LayerNorm                  │
               ↓                      │
        Feed-Forward Network         │
               ↓                      │
               └────── + x ──────────┘
                      │
                      ↓
                    Output
```

---

# 19. Residual Initialization

The GPT-2 implementation also incorporates GPT-2-style residual initialization.

Residual projection weights are initialized with a scale that accounts for the number of residual layers.

The motivation is to prevent residual activations from becoming excessively large as the number of Transformer blocks increases.

Conceptually:

```text
More Transformer blocks
        ↓
More residual additions
        ↓
Potential activation growth
        ↓
Scaled residual initialization
        ↓
More stable initialization
```

This becomes increasingly important when moving toward larger GPT-3-style models.

---

# 20. Final LayerNorm

After all Transformer blocks, the model applies a final LayerNorm:

```text
Transformer Block 1
        ↓
Transformer Block 2
        ↓
...
        ↓
Transformer Block N
        ↓
Final LayerNorm
```

This is another important GPT-2 architectural difference from the original GPT-1 implementation.

The final normalization produces the representation used by the language-model head.

---

# 21. Language Modeling Head

The final hidden representation is projected into vocabulary space.

If:

```text
hidden shape =
[batch_size, sequence_length, d_model]
```

then the language-model head produces:

```text
[batch_size, sequence_length, vocab_size]
```

For the current configuration:

```text
[16, 128, 256]
        ↓
Linear
        ↓
[16, 128, 5000]
```

Each of the 128 positions therefore produces a probability distribution over the 5,000-token vocabulary after applying softmax.

---

# 22. Logits

The model returns **logits**, not probabilities.

For each position:

```text
logits[t]
```

contains one score for every vocabulary token.

Therefore:

```text
logits.shape =
[batch_size, sequence_length, vocab_size]
```

Current configuration:

```text
[16, 128, 5000]
```

The loss function applies the appropriate normalization internally when calculating next-token prediction loss.

---

# 23. Next-Token Prediction

Suppose the input is:

```text
The cat is
```

The model produces predictions for every position:

```text
The  → cat
cat  → is
is   → sleeping
```

The target sequence is shifted by one token.

The model therefore performs many next-token predictions simultaneously.

This is why a single training batch can contain thousands of token-level prediction targets.

---

# 24. Complete Forward Pass

The complete forward pass is:

```text
Input Token IDs
      ↓
Token Embedding
      +
Position Embedding
      ↓
Dropout
      ↓
┌────────────────────────────┐
│ Transformer Block          │
│                            │
│ LayerNorm                  │
│     ↓                      │
│ Causal Self-Attention      │
│     ↓                      │
│ Residual Addition          │
│     ↓                      │
│ LayerNorm                  │
│     ↓                      │
│ Feed-Forward Network       │
│     ↓                      │
│ Residual Addition          │
└────────────────────────────┘
      ↓
Repeat N times
      ↓
Final LayerNorm
      ↓
Language Modeling Head
      ↓
Logits
```

For the current model:

```text
N = 6
```

---

# 25. Tensor Shape Flow

With:

```text
batch_size = 16
context_length = 128
d_model = 256
vocab_size = 5000
```

the forward pass has the following shape transitions:

```text
Input IDs
[16, 128]
     ↓
Token Embedding
[16, 128, 256]
     ↓
Position Embedding
[16, 128, 256]
     ↓
Transformer Blocks
[16, 128, 256]
     ↓
Final LayerNorm
[16, 128, 256]
     ↓
LM Head
[16, 128, 5000]
```

The sequence length remains unchanged throughout the Transformer.

Only the final projection changes the last dimension from:

```text
256 → 5000
```

---

# 26. Parameter Count

The model provides parameter-count functionality so that the architecture can be verified independently of training.

The total parameter count is approximately the sum of:

```text
Token embeddings
+
Position embeddings
+
Transformer block parameters
+
LayerNorm parameters
+
Language-model head parameters
```

For each Transformer block, the major contributors are:

```text
Self-Attention
+
Feed-Forward Network
+
LayerNorm
```

Parameter counting is useful because it provides a sanity check.

If a configuration is expected to contain approximately a certain number of parameters but the implementation reports something significantly different, an architectural component may have been incorrectly implemented.

---

# 27. Parameter Scaling

The major architecture parameters are:

```text
vocab_size
d_model
num_heads
num_layers
ff_hidden_dim
context_length
```

Increasing them affects the model differently.

### Increase `d_model`

Increases the dimensionality of the hidden representation and substantially increases parameter count.

### Increase `num_layers`

Makes the model deeper.

### Increase `num_heads`

Provides more attention heads, while maintaining:

```text
d_model % num_heads == 0
```

### Increase `ff_hidden_dim`

Increases the capacity of the feed-forward sublayers.

### Increase `context_length`

Allows the model to process longer sequences, but increases computational and memory requirements for attention.

---

# 28. Configuration Invariant

The model requires:

```python
d_model % num_heads == 0
```

This ensures that the hidden representation can be evenly divided among attention heads.

For the current configuration:

```text
256 % 4 = 0
```

and therefore:

```text
d_head = 64
```

The configuration validates this requirement during initialization.

---

# 29. GPT2 Model Class

The complete model is exposed through:

```text
src/models/gpt2.py
```

Conceptually:

```python
model = GPT2(config)
```

The model is responsible for:

* Token embeddings
* Position embeddings
* Transformer blocks
* Final LayerNorm
* Language-model projection
* Forward pass
* Parameter counting

Individual architectural components remain modular.

---

# 30. Architectural Testing

The architecture was tested independently before training.

Important tests include:

```text
Configuration validation
        ↓
Embedding shapes
        ↓
Attention shapes
        ↓
Causal masking
        ↓
Feed-forward shapes
        ↓
Transformer block
        ↓
Residual behavior
        ↓
Final LayerNorm
        ↓
GPT-2 forward pass
        ↓
Parameter count
```

The final forward-pass invariant is:

```text
Input:
[batch_size, context_length]

Output:
[batch_size, context_length, vocab_size]
```

For the current configuration:

```text
Input:
[16, 128]

Output:
[16, 128, 5000]
```

---

# 31. GPT-1 → GPT-2 Architectural Evolution

The architectural evolution can be summarized as:

```text
GPT-1
  ↓
Post-LN Transformer
  ↓
GPT-2
  ↓
Pre-LN
  +
Final LayerNorm
  +
Residual initialization
  +
Improved tokenizer/data pipeline
```

The core Transformer remains recognizable, but the changes improve training stability and make the architecture more suitable for scaling.

---

# 32. Why These Changes Matter for GPT-3 Scaling

The GPT-2 implementation is also a bridge toward the next phase of the project.

GPT-3 scaling will increase:

```text
Model depth
Model width
Number of attention heads
Training tokens
Batch size
Training steps
Compute
```

Architectural stability therefore becomes increasingly important.

The GPT-2 implementation introduces the architectural patterns needed to make this transition practical.

The progression is:

```text
GPT-1
Small Transformer
      ↓
GPT-2
Improved Transformer architecture
      ↓
GPT-3
Scaled Transformer architecture
```

---

# 33. Final Architecture Summary

The implemented GPT-2 model can be summarized as:

```text
                 Token IDs
                    │
                    ↓
             Token Embedding
                    │
                    +
            Position Embedding
                    │
                    ↓
                 Dropout
                    │
                    ↓
       ┌──────────────────────────┐
       │     GPT-2 Block × 6      │
       │                          │
       │     LayerNorm            │
       │         ↓                │
       │   Causal Self-Attention  │
       │         ↓                │
       │     Residual Add         │
       │         ↓                │
       │     LayerNorm            │
       │         ↓                │
       │   Feed-Forward Network   │
       │         ↓                │
       │     Residual Add         │
       └──────────────────────────┘
                    │
                    ↓
             Final LayerNorm
                    │
                    ↓
              LM Head
                    │
                    ↓
                  Logits
                    │
                    ↓
       Next-Token Prediction
```

Current architecture:

```text
Vocabulary Size   = 5,000
Context Length    = 128
d_model           = 256
Attention Heads   = 4
Head Dimension    = 64
Transformer Layers= 6
FFN Dimension     = 1,024
Dropout           = 0.1
LayerNorm ε       = 1e-5
```

The model therefore represents a complete GPT-2-style decoder-only Transformer implementation, while remaining small enough for local development and experimentation.
