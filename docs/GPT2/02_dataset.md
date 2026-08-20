# GPT-2 — Dataset

## 1. Overview

The GPT-2 implementation uses a **FineWeb-derived text corpus** as the pretraining dataset.

The purpose of the dataset pipeline is to transform raw documents into a clean, tokenized corpus suitable for autoregressive next-token prediction.

The complete pipeline is:

```text
Raw FineWeb Dataset
        ↓
Document Cleaning
        ↓
Corpus Preprocessing
        ↓
Train / Validation / Test Split
        ↓
GPT-2 Byte-Level BPE Tokenization
        ↓
EOS Token Handling
        ↓
Token Streams
        ↓
Context Windows
        ↓
GPTDataset
        ↓
DataLoader
```

The dataset pipeline is deliberately separated from the model and training code so that each stage can be independently tested and modified.

---

## 2. Why FineWeb?

During the GPT-1 implementation, a small TinyStories dataset was used to make development and debugging practical.

For GPT-2, the objective is closer to general-purpose language-model pretraining.

Therefore, a more diverse, WebText-like corpus is used.

The GPT-2 implementation uses a **small FineWeb-derived subset** rather than attempting to reproduce the enormous dataset used by the original GPT-2 training.

The purpose is to reproduce the **data-processing methodology**, not the original training scale.

### GPT-1

```text
TinyStories
    ↓
Mostly short stories
    ↓
Small-scale language modeling
```

### GPT-2

```text
FineWeb-derived corpus
    ↓
Diverse web-style documents
    ↓
General language modeling
```

This change is important because the model is intended to learn more general language patterns rather than primarily the structure of short children's stories.

---

## 3. Dataset Source

The processed corpus used by the implementation is represented by:

```text
data/processed/fineweb_10k_clean.txt
```

The dataset is intentionally kept small enough to allow development and experimentation on local hardware.

The corpus contains approximately:

```text
10,000 documents
```

after preprocessing and filtering.

This is **not equivalent to the scale of the original GPT-2 training corpus**.

The smaller dataset is used because the primary objective of this project is implementation and experimentation.

---

## 4. Document Cleaning

Raw web data can contain unwanted formatting, malformed text, empty documents, and other artifacts.

The preprocessing stage therefore performs document cleaning before tokenization.

Conceptually:

```text
Raw document
     ↓
Remove unwanted content
     ↓
Normalize / clean text
     ↓
Validate document
     ↓
Keep or discard
```

Cleaning is performed before the train/validation/test split so that the resulting dataset contains only documents suitable for model training.

The cleaning logic is kept separate from tokenization.

This separation allows tokenizer behavior to be tested independently from dataset cleaning.

---

## 5. Document Representation

At the preprocessing stage, the dataset is represented as a collection of documents:

```python
documents: list[str]
```

Each document is an independent text sample.

For example:

```text
Document 1
Document 2
Document 3
...
Document N
```

Documents are not immediately concatenated into one global sequence.

This is important because document boundaries must be preserved until the appropriate stage of the pipeline.

---

## 6. Train / Validation / Test Split

After preprocessing, the documents are divided into three subsets:

```text
                 All Documents
                       │
          ┌────────────┼────────────┐
          ↓            ↓            ↓
        Train          Val         Test
```

### Training set

Used to update model parameters.

```text
Train → forward → loss → backward → optimizer
```

### Validation set

Used to monitor generalization during development.

It is not used to update model parameters.

```text
Validation → loss → model monitoring
```

### Test set

Used for final held-out evaluation after training.

```text
Test → final loss → perplexity
```

The split is implemented separately in:

```text
src/datasets/split.py
```

This keeps dataset splitting independent from model training.

---

## 7. Tokenization

After splitting, each document is passed through the GPT-2 tokenizer.

The tokenization process is:

```text
Document
   ↓
Byte Encoding
   ↓
GPT-2 Pre-tokenization
   ↓
BPE
   ↓
Token IDs
```

The tokenizer implementation is described in detail in:

```text
docs/gpt2/tokenizer.md
```

The important dataset-level result is that each document becomes a list of integer token IDs:

```python
[
    [token_1, token_2, token_3, ...],
    [token_1, token_2, token_3, ...],
    ...
]
```

---

## 8. End-of-Text Token

Document boundaries are explicitly represented using:

```text
<|endoftext|>
```

This is an important difference from simply concatenating documents without any boundary information.

Conceptually, the corpus becomes:

```text
[doc1 tokens] [EOS]
[doc2 tokens] [EOS]
[doc3 tokens] [EOS]
...
```

The EOS token tells the model that one document has ended and another document begins.

This prevents the model from interpreting the last token of one document and the first token of the next document as an ordinary continuous sentence.

---

## 9. Token Streams

After tokenization, each split is converted into a token stream.

For example:

```text
Train documents
      ↓
Tokenized documents
      ↓
EOS appended to documents
      ↓
Flattened token stream
```

Conceptually:

```text
doc1₁ doc1₂ doc1₃ EOS
doc2₁ doc2₂ doc2₃ EOS
doc3₁ doc3₂ doc3₃ EOS
```

becomes:

```text
doc1₁ doc1₂ doc1₃ EOS doc2₁ doc2₂ doc2₃ EOS doc3₁ ...
```

The same operation is performed independently for:

```text
train_ids → train_stream
val_ids   → val_stream
test_ids  → test_stream
```

This prevents information from the validation or test datasets from entering the training stream.

---

## 10. Why Preserve Document Boundaries?

A language model is trained on sequences, but the original corpus consists of separate documents.

Without an EOS token:

```text
Document A
       ↓
Document B
```

could become:

```text
...end of Document A beginning of Document B...
```

with no indication that the boundary exists.

With EOS:

```text
...Document A... EOS ...Document B...
```

the model can learn that the previous document has ended.

This is especially important when constructing context windows across a large corpus.

---

## 11. Context Windows

The token stream is converted into fixed-length training examples.

The GPT-2 configuration currently uses:

```python
context_length = 128
stride = 128
```

For a context length of 128, a training sample contains:

```text
Input:
t₀ t₁ t₂ ... t₁₂₇

Target:
t₁ t₂ t₃ ... t₁₂₈
```

The target sequence is shifted by one token relative to the input.

Therefore:

```text
input[i] → target[i]
```

represents:

```text
predict the next token
```

---

## 12. Window Indexing

Window construction is separated into a utility function:

```text
src/utils/window.py
```

The purpose of the window index builder is to determine where each context window begins.

For:

```text
context_length = 128
stride = 128
```

the windows are non-overlapping.

Conceptually:

```text
Token stream

0 ───────── 127
            ↑
          window 1

128 ─────── 255
            ↑
          window 2

256 ─────── 383
            ↑
          window 3
```

If a different stride is selected, windows can overlap.

For example:

```text
context_length = 128
stride = 64
```

produces:

```text
0 ───────── 127
64 ───────── 191
128 ──────── 255
```

This allows the amount of overlap between training examples to be controlled.

---

## 13. Short Documents

Documents shorter than the required context length are handled during the token-stream/window construction process.

The implementation does not create an incomplete input tensor because the model expects a fixed context length.

The objective is to ensure that every sample returned by `GPTDataset` has:

```text
input shape  = [context_length]
target shape = [context_length]
```

This guarantees consistent batch shapes.

---

## 14. Input / Target Alignment

Input and target construction is isolated into:

```text
src/utils/alignment.py
```

For:

```text
tokens = [10, 20, 30, 40, 50]
context_length = 4
```

the sample becomes:

```text
Input:
[10, 20, 30, 40]

Target:
[20, 30, 40, 50]
```

This is the fundamental training relationship for the autoregressive language model.

The model therefore learns:

```text
10 → 20
20 → 30
30 → 40
40 → 50
```

while causal attention prevents the model from looking at future target tokens.

---

## 15. GPTDataset

The context-window samples are exposed through a PyTorch `Dataset`.

The implementation is located in:

```text
src/datasets/dataset.py
```

The dataset receives:

```python
token_stream
context_length
stride
```

and internally creates the window indices.

Conceptually:

```text
GPTDataset
    │
    ├── token_stream
    ├── context_length
    ├── stride
    │
    ↓
window_indices
    ↓
__getitem__()
    ↓
input_ids
target_ids
```

Each item returned by the dataset is:

```python
(
    input_ids,
    target_ids
)
```

with both represented as PyTorch tensors.

---

## 16. DataLoader

The `GPTDataset` is passed to the common DataLoader factory:

```text
src/datasets/dataloader.py
```

The DataLoader is responsible for batching individual samples.

With:

```python
batch_size = 16
context_length = 128
```

the expected batch shape is:

```text
input_ids:
[16, 128]

target_ids:
[16, 128]
```

This was explicitly tested during the implementation.

The training DataLoader uses:

```python
shuffle=True
```

while validation and test DataLoaders use:

```python
shuffle=False
```

---

## 17. Complete Data Pipeline

The complete GPT-2 dataset pipeline can therefore be summarized as:

```text
FineWeb-derived corpus
          ↓
      Cleaning
          ↓
    Preprocessing
          ↓
 Train / Val / Test
          ↓
    GPT-2 Tokenizer
          ↓
     Token IDs
          ↓
       EOS
          ↓
    Token Streams
          ↓
  Window Index Builder
          ↓
 Context Windows
          ↓
 Input / Target Alignment
          ↓
      GPTDataset
          ↓
      DataLoader
          ↓
      GPT-2 Training
```

---

## 18. Implementation Modules

The main dataset-related components are:

```text
src/datasets/
├── download.py
├── clean.py
├── preprocess.py
├── split.py
├── tokenize.py
├── dataset.py
├── dataloader.py
└── data_pipeline.py
```

Supporting utilities include:

```text
src/utils/
├── window.py
└── alignment.py
```

### Responsibilities

| Module             | Responsibility                          |
| ------------------ | --------------------------------------- |
| `download.py`      | Dataset acquisition                     |
| `clean.py`         | Document cleaning                       |
| `preprocess.py`    | Corpus preprocessing                    |
| `split.py`         | Train/validation/test split             |
| `tokenize.py`      | Document tokenization and token streams |
| `dataset.py`       | Context-window Dataset                  |
| `dataloader.py`    | PyTorch DataLoader creation             |
| `data_pipeline.py` | End-to-end dataset pipeline             |
| `window.py`        | Window index construction               |
| `alignment.py`     | Input/target construction               |

---

## 19. Data Pipeline Testing

The pipeline was developed incrementally with tests for the individual components.

Important tests include:

```text
Document preprocessing
        ↓
Train/Val/Test split
        ↓
Tokenization
        ↓
EOS handling
        ↓
Window index construction
        ↓
Input/target alignment
        ↓
GPTDataset
        ↓
Batch shape
        ↓
Train/Val/Test DataLoaders
```

The final batch-shape expectation is:

```text
Batch size       = 16
Context length   = 128

Input:
[16, 128]

Target:
[16, 128]
```

This ensures that the dataset pipeline produces tensors compatible with the GPT-2 model.

---

## 20. GPT-1 → GPT-2 Dataset Changes

| Area                 | GPT-1              | GPT-2                                  |
| -------------------- | ------------------ | -------------------------------------- |
| Corpus               | TinyStories        | FineWeb-derived corpus                 |
| Dataset objective    | Small-scale LM     | General language modeling              |
| Documents            | Short stories      | Web-style documents                    |
| Split                | Train / Validation | Train / Validation / Test              |
| Tokenization         | GPT-1 tokenizer    | GPT-2 byte-level BPE                   |
| EOS                  | Basic handling     | Explicit document boundary             |
| Token representation | Tokenized samples  | Token streams                          |
| Context construction | Fixed samples      | Context windows                        |
| Window stride        | Limited            | Explicit configurable stride           |
| Input/target         | Shifted tokens     | Dedicated alignment utility            |
| Dataset              | GPT-1 Dataset      | `GPTDataset`                           |
| Batching             | DataLoader         | DataLoader with train/val/test loaders |
| Testing              | Basic              | Component + shape tests                |

---

## 21. Key Takeaway

The main dataset improvement in the GPT-2 implementation is the transition from a simple collection of pre-built samples to a structured **document → token stream → context window** pipeline.

The important abstraction is:

```text
Documents
    ↓
Token Streams
    ↓
Windows
    ↓
Input / Target Pairs
    ↓
Batches
```

