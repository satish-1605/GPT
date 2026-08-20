# GPT-2 — Tokenizer

## 1. Overview

The GPT-2 implementation uses a **byte-level Byte Pair Encoding (BPE)** tokenizer inspired by the tokenizer design used by GPT-2.

The tokenizer converts raw text into integer token IDs that can be consumed by the GPT-2 model.

The complete pipeline is:

```text
Raw Text
   ↓
Byte Encoding
   ↓
GPT-2 Pre-tokenization
   ↓
BPE Merge Operations
   ↓
Vocabulary Lookup
   ↓
Token IDs
```

For decoding:

```text
Token IDs
   ↓
Vocabulary Lookup
   ↓
Byte Sequences
   ↓
Byte Decoding
   ↓
Text
```

The tokenizer is implemented from scratch rather than relying on a pre-built GPT-2 tokenizer.

---

## 2. Why Byte-Level BPE?

The GPT-1 implementation used a simpler BPE-based tokenizer.

For GPT-2, the tokenizer is improved by operating at the **byte level**.

The main motivation is to avoid requiring every possible Unicode character or word to exist directly in the vocabulary.

Instead, arbitrary text can first be represented using bytes.

Conceptually:

```text
Text
 ↓
UTF-8 bytes
 ↓
Byte-level representation
 ↓
BPE
 ↓
Tokens
```

This gives the tokenizer a much more general representation of text.

---

## 3. Tokenizer Components

The tokenizer is divided into several components:

```text
src/tokenizer/
├── byte_encoder.py
├── pretokenizer.py
├── vocabulary.py
├── bpe.py
└── tokenizer.py
```

The main responsibilities are:

| Component          | Responsibility                                       |
| ------------------ | ---------------------------------------------------- |
| `ByteEncoder`      | Converts bytes into reversible token representations |
| `GPT2PreTokenizer` | Splits raw text using GPT-2-style regex rules        |
| `Vocabulary`       | Maintains token ↔ ID mappings                        |
| `BPE`              | Learns and applies merge operations                  |
| `BPETokenizer`     | Provides the complete encode/decode interface        |

The design keeps each tokenizer component independently testable.

---

# 4. Byte Encoding

## 4.1 Problem

Raw text contains Unicode characters.

For example:

```text
Hello
भारत
é
😊
```

These characters cannot simply be assumed to correspond to one byte each.

UTF-8 represents characters using one or more bytes.

Therefore, the tokenizer first converts text into its byte representation.

---

## 4.2 Byte-Level Representation

For example:

```text
"Hello"
```

is represented using UTF-8 bytes.

The tokenizer then maps these bytes to a reversible internal representation.

Conceptually:

```text
"Hello"
   ↓
UTF-8 bytes
   ↓
Byte encoder
   ↓
Byte-level symbols
```

The important property is **reversibility**.

The transformation must satisfy:

```text
decode(encode(text)) == text
```

for valid input text.

---

## 4.3 Why a Reversible Byte Mapping?

The tokenizer should not lose information during preprocessing.

If an unknown Unicode character appears in the dataset, the tokenizer should still be able to represent it.

Instead of requiring:

```text
"😊" → dedicated vocabulary token
```

the tokenizer can represent the underlying bytes.

This gives the tokenizer coverage over arbitrary UTF-8 text.

---

# 5. GPT-2 Pre-tokenization

After byte-level preparation, the tokenizer applies GPT-2-style pre-tokenization.

The purpose of pre-tokenization is to divide raw text into meaningful pieces before applying BPE.

Conceptually:

```text
Raw text
   ↓
GPT-2 regex
   ↓
Pre-tokenized pieces
   ↓
BPE
```

The pre-tokenizer is implemented separately as:

```text
GPT2PreTokenizer
```

This allows the GPT-2 regex behavior to be tested independently from the BPE implementation.

---

## 5.1 Why Pre-tokenization?

Consider:

```text
The cat doesn't run quickly.
```

The tokenizer should preserve useful linguistic structure rather than treating the entire sentence as one BPE sequence.

The pre-tokenization stage identifies pieces involving:

* Letters
* Numbers
* Punctuation
* Whitespace
* Contractions

The BPE algorithm then operates on these pieces.

---

# 6. Byte-Level BPE

Byte Pair Encoding progressively combines frequently occurring symbols.

The basic process is:

```text
Initial symbols
      ↓
Find most frequent pair
      ↓
Merge pair
      ↓
Update vocabulary
      ↓
Repeat
```

For example, suppose the token sequence contains:

```text
l o w e r
l o w e s t
```

Frequently occurring pairs may be merged:

```text
l + o → lo
```

followed by:

```text
lo + w → low
```

Eventually, frequently occurring sequences become larger vocabulary tokens.

The actual merge operations are learned from the training corpus.

---

# 7. BPE Training

The BPE trainer learns a fixed number of vocabulary entries from the training corpus.

In the GPT-2 implementation, the vocabulary size is configurable.

For example:

```python
VOCAB_SIZE = 5000
```

The training process is conceptually:

```text
Training corpus
      ↓
Pre-tokenization
      ↓
Byte encoding
      ↓
Initial vocabulary
      ↓
Count symbol pairs
      ↓
Select most frequent pair
      ↓
Merge pair
      ↓
Repeat
      ↓
Vocabulary + merges
```

The resulting artifacts are saved and reused during model training and inference.

---

# 8. Vocabulary

The vocabulary maintains a mapping between token representations and integer IDs.

Conceptually:

```text
Token → ID

token_a → 0
token_b → 1
token_c → 2
...
```

The reverse mapping is also maintained:

```text
ID → Token
```

This is required during decoding.

The vocabulary therefore supports:

```text
encode:
token → integer ID

decode:
integer ID → token
```

---

# 9. Merge Rules

The BPE tokenizer maintains an ordered list of merge operations.

Conceptually:

```text
merge_1
merge_2
merge_3
...
merge_N
```

The order is important because earlier learned merges have higher priority.

During tokenization, the BPE algorithm repeatedly applies the appropriate learned merge rules.

The resulting token sequence is then converted into vocabulary IDs.

---

# 10. Special End-of-Text Token

The GPT-2 implementation explicitly uses:

```text
<|endoftext|>
```

as the document boundary token.

This token is important for the dataset pipeline.

Documents are represented conceptually as:

```text
Document 1
<|endoftext|>
Document 2
<|endoftext|>
Document 3
<|endoftext|>
```

After tokenization:

```text
[doc1 tokens] EOS [doc2 tokens] EOS [doc3 tokens] EOS
```

This allows the model to learn where one document ends and another begins.

---

# 11. Encoding

The public tokenizer interface provides an `encode()` operation.

Conceptually:

```python
token_ids = tokenizer.encode(text)
```

The process is:

```text
Input text
   ↓
GPT-2 pre-tokenization
   ↓
Byte encoding
   ↓
BPE
   ↓
Vocabulary lookup
   ↓
Token IDs
```

Example:

```text
Input:
"Hello world"

Output:
[token_id_1, token_id_2, ...]
```

The actual IDs depend on the vocabulary and learned merge rules.

---

# 12. Decoding

The reverse operation converts token IDs back into text.

```python
text = tokenizer.decode(token_ids)
```

The process is:

```text
Token IDs
   ↓
Vocabulary lookup
   ↓
BPE token reconstruction
   ↓
Byte representation
   ↓
UTF-8 decoding
   ↓
Text
```

The desired property is:

```text
decode(encode(text)) == text
```

for supported text.

This round-trip property is one of the most important tokenizer tests.

---

# 13. Tokenizer Persistence

The trained tokenizer is saved to disk so that it does not need to be retrained every time the model is executed.

The tokenizer artifacts include:

```text
artifacts/tokenizer/
├── vocab.json
├── merges.txt
└── config.json
```

### `vocab.json`

Stores the vocabulary mapping.

```text
token → token ID
```

### `merges.txt`

Stores the learned BPE merge operations.

### `config.json`

Stores tokenizer configuration and metadata required to reconstruct the tokenizer.

---

# 14. Loading a Trained Tokenizer

The tokenizer can be reconstructed from the saved artifacts:

```python
tokenizer = BPETokenizer.from_pretrained(
    config.load_dir
)
```

This allows the same tokenizer to be shared across:

```text
Training
   ↓
Validation
   ↓
Testing
   ↓
Inference
```

Using the same tokenizer everywhere is critical.

The model's vocabulary size and token IDs must remain consistent between training and inference.

---

# 15. Tokenizer and Dataset Integration

The tokenizer is used after the dataset has been cleaned and split.

The pipeline is:

```text
FineWeb documents
       ↓
Preprocessing
       ↓
Train / Val / Test
       ↓
CorpusTokenizer
       ↓
BPETokenizer
       ↓
Tokenized documents
       ↓
EOS
       ↓
Token streams
```

The `CorpusTokenizer` acts as the bridge between the dataset pipeline and the tokenizer.

Conceptually:

```text
CorpusTokenizer
      │
      ├── tokenize_documents()
      │
      └── build_token_stream()
```

This keeps corpus-level processing separate from the lower-level BPE tokenizer.

---

# 16. Document-Level Tokenization

Each document is tokenized independently.

Conceptually:

```python
tokenized_documents = [
    tokenizer.encode(document_1),
    tokenizer.encode(document_2),
    tokenizer.encode(document_3),
]
```

The EOS token is then used to preserve document boundaries when constructing the token stream.

The important design decision is that we **do not simply throw away leftover tokens from one document because a context window ended**.

Instead, the token stream and EOS boundaries allow the context-window stage to use the available token sequence efficiently while retaining document-boundary information.

---

# 17. Vocabulary Size

Vocabulary size is a configurable tokenizer parameter.

For example:

```python
VOCAB_SIZE = 5000
```

A larger vocabulary generally allows more frequent sequences to be represented as individual tokens.

However, increasing vocabulary size also increases:

```text
Embedding parameters
+
Output projection parameters
```

and therefore affects model size.

Vocabulary size is consequently both a tokenizer and model-design consideration.

---

# 18. Tokenizer Testing

The tokenizer was developed with independent tests for its major components.

Important test categories include:

```text
Byte Encoding
     ↓
Byte Decoding
     ↓
Pre-tokenization
     ↓
Vocabulary
     ↓
BPE merge behavior
     ↓
Encode
     ↓
Decode
     ↓
Round-trip
     ↓
Persistence
     ↓
Reload
```

The tests verify that the individual tokenizer components work correctly before the tokenizer is used by the GPT-2 training pipeline.

---

# 19. Important Invariants

Several properties must remain true throughout the implementation.

### Round-trip correctness

```text
decode(encode(text)) == text
```

### Vocabulary consistency

```text
token → ID → token
```

must be deterministic.

### Deterministic inference

A tokenizer loaded from the same artifacts must produce the same token IDs.

### Stable vocabulary

Once training starts, the tokenizer vocabulary and merge rules must remain fixed.

The model is trained against a specific mapping:

```text
token → token ID
```

Changing the tokenizer after model training would make the model's learned embeddings incompatible with the new vocabulary.

---

# 20. GPT-1 → GPT-2 Tokenizer Changes

| Area             | GPT-1                        | GPT-2                                      |           |             |
| ---------------- | ---------------------------- | ------------------------------------------ | --------- | ----------- |
| Basic algorithm  | BPE                          | Byte-level BPE                             |           |             |
| Byte encoding    | Limited / not central        | Explicit byte encoder                      |           |             |
| Pre-tokenization | Simpler                      | GPT-2-style regex                          |           |             |
| Unicode handling | More dependent on vocabulary | Byte-level representation                  |           |             |
| BPE merges       | Yes                          | Yes                                        |           |             |
| Vocabulary       | Learned                      | Learned                                    |           |             |
| EOS              | Supported                    | Explicit `<                                | endoftext | >` boundary |
| Persistence      | Vocabulary/merges            | Vocabulary + merges + config               |           |             |
| Architecture     | Monolithic tokenizer flow    | Modular components                         |           |             |
| Testing          | Tokenizer tests              | Component + round-trip + persistence tests |           |             |

---

# 21. Why the GPT-2 Tokenizer Matters

The tokenizer is not merely a preprocessing utility.

It defines the interface between raw language and the neural network.

The model does not directly see:

```text
"The cat is sleeping."
```

It sees:

```text
[token_id_1, token_id_2, token_id_3, ...]
```

Therefore:

```text
Raw Text
   ↓
Tokenizer
   ↓
Token IDs
   ↓
GPT-2
```

The quality and consistency of this transformation directly affects the model's ability to learn language patterns.

---

# 22. Key Takeaway

The GPT-2 tokenizer improves the GPT-1 tokenizer by introducing a more complete **byte-level BPE pipeline**:

```text
Text
 ↓
Byte Encoding
 ↓
GPT-2 Pre-tokenization
 ↓
BPE
 ↓
Vocabulary
 ↓
Token IDs
```

The tokenizer is:

* Byte-level
* BPE-based
* GPT-2-style pre-tokenized
* EOS-aware
* Persistable
* Reloadable
* Independently tested

This tokenizer provides the foundation for the GPT-2 dataset pipeline and ensures that the same token-to-ID mapping is used consistently during:

```text
Pretraining
Validation
Testing
Inference
```
