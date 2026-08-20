# GPT-2 — Data Pipeline

## 1. Overview

The GPT-2 data pipeline converts a collection of raw text documents into batches of input and target token sequences suitable for autoregressive language-model training.

The complete pipeline is:

```text
Raw Documents
     ↓
Preprocessing
     ↓
Train / Validation / Test Split
     ↓
Document Tokenization
     ↓
EOS Handling
     ↓
Token Streams
     ↓
Context Window Indexing
     ↓
Input / Target Alignment
     ↓
GPTDataset
     ↓
DataLoader
     ↓
Batches
```

The pipeline is implemented as independent components so that each stage can be tested separately.

---

## 2. Design Goals

The GPT-2 data pipeline was designed around the following goals:

1. Preserve document boundaries.
2. Use the same tokenizer for training, validation, testing, and inference.
3. Efficiently construct fixed-length context windows.
4. Generate correctly aligned input/target pairs.
5. Produce fixed-size batches for the model.
6. Keep dataset preparation separate from model and training logic.
7. Make each component independently testable.

---

# 3. Dataset Pipeline Architecture

The main dataset modules are:

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

Supporting utilities:

```text
src/utils/
├── window.py
└── alignment.py
```

The responsibilities are:

| Module             | Responsibility                                      |
| ------------------ | --------------------------------------------------- |
| `download.py`      | Dataset acquisition                                 |
| `clean.py`         | Text cleaning                                       |
| `preprocess.py`    | Corpus preprocessing                                |
| `split.py`         | Train/validation/test split                         |
| `tokenize.py`      | Document tokenization and token-stream construction |
| `dataset.py`       | PyTorch dataset                                     |
| `dataloader.py`    | Batch creation                                      |
| `data_pipeline.py` | End-to-end pipeline                                 |
| `window.py`        | Context-window index generation                     |
| `alignment.py`     | Input/target alignment                              |

---

# 4. Preprocessing

The first stage converts the raw dataset into a collection of usable documents.

Conceptually:

```text
Raw Dataset
    ↓
Cleaning
    ↓
Filtering
    ↓
Processed Documents
```

The preprocessing stage is intentionally independent of tokenization.

This allows text-cleaning behavior to be tested without involving the BPE implementation.

The resulting representation is:

```python
documents: list[str]
```

Each element represents one document.

---

# 5. Train / Validation / Test Split

After preprocessing, the documents are divided into three independent subsets:

```text
                    Documents
                       │
          ┌────────────┼────────────┐
          ↓            ↓            ↓
        Train          Val         Test
```

The split is performed before tokenization.

This ensures that the training, validation, and test corpora remain independent.

### Training set

Used for parameter updates.

### Validation set

Used to monitor model generalization during training.

### Test set

Used for final evaluation.

The split functionality is implemented in:

```text
src/datasets/split.py
```

---

# 6. Document Tokenization

Each document is passed through the GPT-2 byte-level BPE tokenizer.

The `CorpusTokenizer` provides the corpus-level interface:

```text
Documents
    ↓
CorpusTokenizer
    ↓
BPETokenizer
    ↓
Token IDs
```

Conceptually:

```python
train_ids = corpus_tokenizer.tokenize_documents(train_docs)
val_ids = corpus_tokenizer.tokenize_documents(val_docs)
test_ids = corpus_tokenizer.tokenize_documents(test_docs)
```

The result is a list of tokenized documents:

```text
[
    [token_1, token_2, token_3, ...],
    [token_1, token_2, token_3, ...],
    ...
]
```

---

# 7. EOS Handling

Each document is terminated using the special:

```text
<|endoftext|>
```

token.

The corpus therefore has the conceptual structure:

```text
[Document 1 tokens] EOS
[Document 2 tokens] EOS
[Document 3 tokens] EOS
```

This preserves document boundaries when the documents are eventually combined into token streams.

The important design decision is that the EOS token is added at the **document level**, before the documents are flattened.

This allows the model to observe explicit document boundaries.

---

# 8. Token Streams

After tokenization, the tokenized documents are converted into a single token stream for each dataset split.

For example:

```text
Train documents
      ↓
Train tokenized documents
      ↓
Flatten
      ↓
Train token stream
```

The same operation is performed independently for validation and test data:

```text
train_ids → train_stream
val_ids   → val_stream
test_ids  → test_stream
```

The resulting representation is:

```python
train_stream: list[int]
val_stream: list[int]
test_stream: list[int]
```

The streams are kept separate to prevent data leakage.

---

# 9. Why Use Token Streams?

The language model ultimately learns from sequences of tokens rather than individual documents.

A token stream allows us to construct context windows systematically.

For example:

```text
Token stream:

t0 t1 t2 t3 t4 t5 t6 t7 t8 ...
```

can be transformed into:

```text
Window 1:
t0 ... t127

Window 2:
t128 ... t255

Window 3:
t256 ... t383
```

The token-stream representation therefore provides a clean interface between tokenization and context-window construction.

---

# 10. Context Length

The GPT-2 implementation uses a configurable context length.

Current configuration:

```python
context_length = 128
```

This means each training example contains 128 input tokens.

The model therefore receives:

```text
[batch_size, context_length]
```

as its input shape.

For the current configuration:

```text
[16, 128]
```

---

# 11. Stride

The stride controls how far the window moves after each training example.

Current configuration:

```python
stride = 128
```

With:

```text
context_length = 128
stride = 128
```

the windows are non-overlapping:

```text
0 ───────────── 127
128 ─────────── 255
256 ─────────── 383
```

A smaller stride would produce overlapping windows.

For example:

```text
context_length = 128
stride = 64
```

would produce:

```text
0 ───────────── 127
64 ───────────── 191
128 ──────────── 255
```

Stride is therefore an important control over how densely the token stream is sampled.

---

# 12. Window Index Builder

Instead of storing every context window explicitly, the implementation first creates the starting indices for valid windows.

This functionality is implemented in:

```text
src/utils/window.py
```

The utility:

```python
build_window_indices(
    token_stream,
    context_length,
    stride,
)
```

returns the valid starting positions.

Conceptually:

```text
Token stream
     ↓
Window index builder
     ↓
[0, 128, 256, 384, ...]
```

The dataset can then use these indices to construct samples when required.

This avoids unnecessarily duplicating the entire token stream.

---

# 13. Input / Target Construction

For autoregressive language modeling, the target sequence is shifted by one token.

For example:

```text
Token sequence:

[10, 20, 30, 40, 50]
```

produces:

```text
Input:
[10, 20, 30, 40]

Target:
[20, 30, 40, 50]
```

The implementation isolates this operation in:

```text
src/utils/alignment.py
```

using:

```python
build_input_target(...)
```

This utility was tested independently before integrating it into `GPTDataset`.

---

# 14. Why the Shift Is Required

The model is trained to answer:

> Given the tokens seen so far, what is the next token?

For:

```text
Input:
The cat is

Target:
cat is sleeping
```

the model performs:

```text
The  → cat
cat  → is
is   → sleeping
```

The Transformer uses causal masking so that prediction at position `t` cannot access future tokens.

---

# 15. GPTDataset

The context-window samples are exposed through a PyTorch `Dataset`.

Implementation:

```text
src/datasets/dataset.py
```

The dataset receives:

```python
GPTDataset(
    token_stream=token_stream,
    context_length=context_length,
    stride=stride,
)
```

Internally it:

1. Builds window indices.
2. Selects the requested window.
3. Constructs input/target sequences.
4. Converts them into PyTorch tensors.

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

---

# 16. Dataset Output

Each call to:

```python
dataset[idx]
```

returns:

```python
(
    input_ids,
    target_ids
)
```

Both are `torch.long` tensors.

For:

```text
context_length = 128
```

the expected individual sample shapes are:

```text
input_ids:
[128]

target_ids:
[128]
```

---

# 17. DataLoader

The PyTorch DataLoader converts individual dataset samples into batches.

The common DataLoader factory is implemented in:

```text
src/datasets/dataloader.py
```

The interface remains simple:

```python
create_dataloader(
    dataset,
    batch_size,
    shuffle,
    num_workers,
)
```

This allows the same DataLoader implementation to be reused for:

```text
Training
Validation
Testing
```

---

# 18. Train / Validation / Test Loaders

The complete loader creation is handled by:

```text
src/datasets/data_pipeline.py
```

The pipeline creates:

```text
train_loader
val_loader
test_loader
```

### Training

```python
shuffle=True
```

The training samples are shuffled so that batches are not always presented in the same order.

### Validation

```python
shuffle=False
```

### Test

```python
shuffle=False
```

The validation and test loaders are deterministic in ordering.

---

# 19. Complete Data Pipeline Implementation

The high-level pipeline can be represented as:

```text
FineWeb Dataset
       ↓
DatasetPreprocessor
       ↓
documents
       ↓
train_val_test_split()
       ↓
┌──────────┬──────────┬──────────┐
↓          ↓          ↓
train_docs val_docs  test_docs
↓          ↓          ↓
CorpusTokenizer
↓          ↓          ↓
train_ids  val_ids   test_ids
↓          ↓          ↓
build_token_stream()
↓          ↓          ↓
train_stream val_stream test_stream
↓          ↓          ↓
GPTDataset
↓          ↓          ↓
train_loader val_loader test_loader
```

---

# 20. `data_pipeline.py`

The high-level pipeline combines the individual components.

The flow is approximately:

```python
def get_train_val_test_loaders(config):
    ...
```

The function:

1. Loads the trained tokenizer.
2. Loads and preprocesses the corpus.
3. Splits documents.
4. Tokenizes each split.
5. Builds token streams.
6. Creates datasets.
7. Creates DataLoaders.
8. Returns the three loaders.

The training code therefore does not need to know the internal details of dataset construction.

It simply receives:

```python
train_loader, val_loader, test_loader
```

---

# 21. Batch Shape

The current GPT-2 configuration uses:

```python
batch_size = 16
context_length = 128
```

Therefore the expected batch shape is:

```text
Input:
torch.Size([16, 128])

Target:
torch.Size([16, 128])
```

This was explicitly tested.

The dimensions represent:

```text
16  → number of sequences in the batch
128 → tokens per sequence
```

Therefore:

```text
batch = 16 sequences
```

does **not** mean 16 tokens.

It means the model processes 16 separate context sequences simultaneously.

---

# 22. Relationship Between Batch and Training Step

One optimizer step corresponds to one batch processed by the training loop.

For example:

```text
Batch size = 16
```

and:

```text
Global step = 1
```

means:

```text
16 sequences
      ↓
forward pass
      ↓
loss
      ↓
backward pass
      ↓
optimizer update
      ↓
global_step = 1
```

Therefore:

```text
1 batch ≈ 1 optimizer step
```

in the current training implementation.

The number of tokens processed in one optimizer step is approximately:

```text
batch_size × context_length
```

For the current configuration:

```text
16 × 128 = 2,048 tokens
```

So one training step processes:

```text
2,048 token positions
```

before the optimizer update.

---

# 23. Batch Shape Testing

The pipeline includes explicit shape validation.

The expected invariant is:

```text
input_ids.shape
    == (batch_size, context_length)

target_ids.shape
    == (batch_size, context_length)
```

For the current configuration:

```text
input_ids.shape  == (16, 128)
target_ids.shape == (16, 128)
```

This test is useful because shape errors can otherwise appear much later inside the Transformer.

Testing at the DataLoader boundary catches such problems early.

---

# 24. Pipeline Testing Strategy

The data pipeline was developed incrementally.

The major test stages were:

```text
Preprocessing
     ↓
Train / Val / Test split
     ↓
Document tokenization
     ↓
EOS handling
     ↓
Token streams
     ↓
Window indices
     ↓
Input / target alignment
     ↓
GPTDataset
     ↓
Batch shape
     ↓
Train / Val / Test loaders
```

Each stage was tested before moving to the next stage.

This makes debugging significantly easier than testing only the complete pipeline.

---

# 25. Data Leakage Prevention

The pipeline maintains separate token streams for:

```text
Train
Validation
Test
```

The flow is:

```text
Documents
   ↓
Split
   ↓
┌───────────────┐
│               │
Train          Val          Test
│               │            │
↓               ↓            ↓
train_stream  val_stream   test_stream
```

The training stream is never constructed from validation or test documents.

Similarly, validation and test data are never used for optimizer updates.

This separation is important for obtaining meaningful evaluation results.

---

# 26. Why the Pipeline Is Separate From Training

The training loop should not be responsible for:

* Loading raw data
* Cleaning documents
* Splitting documents
* Tokenizing documents
* Building windows
* Constructing datasets

Instead:

```text
Data Pipeline
      ↓
DataLoader
      ↓
Training Loop
```

This separation gives the project a cleaner architecture:

```text
Dataset Layer
      ↓
Model Layer
      ↓
Training Layer
      ↓
Evaluation Layer
```

Each layer can therefore evolve independently.

---

# 27. GPT-1 → GPT-2 Data Pipeline Changes

| Component            | GPT-1              | GPT-2                         |
| -------------------- | ------------------ | ----------------------------- |
| Dataset              | TinyStories        | FineWeb-derived corpus        |
| Data representation  | Tokenized samples  | Token streams                 |
| Splits               | Train / Validation | Train / Validation / Test     |
| Tokenizer            | GPT-1 tokenizer    | GPT-2 byte-level BPE          |
| EOS handling         | Simpler            | Explicit document boundary    |
| Context construction | Fixed blocks       | Configurable windows          |
| Stride               | Limited            | Explicit configurable stride  |
| Alignment            | Dataset-level      | Dedicated utility             |
| Dataset              | GPT Dataset        | `GPTDataset`                  |
| DataLoader           | Basic              | Train/Val/Test loaders        |
| Testing              | Basic              | Component + batch-shape tests |

---

# 28. Key Design Decisions

### 1. Split before tokenization

This keeps the dataset boundaries explicit and prevents accidental mixing of documents.

### 2. Add EOS at the document level

This preserves document boundaries after documents are flattened into token streams.

### 3. Use a token stream

This makes context-window construction independent of document storage.

### 4. Separate window indexing from dataset logic

`build_window_indices()` handles where windows start, while `GPTDataset` handles retrieving samples.

### 5. Separate input/target alignment

`build_input_target()` has one responsibility: construct shifted input/target sequences.

### 6. Keep DataLoader creation generic

The same DataLoader factory can be used by training, validation, and testing.

---

# 29. Final Pipeline

The final GPT-2 data pipeline is:

```text
                FineWeb Corpus
                      │
                      ↓
               Preprocessing
                      │
                      ↓
             Train / Val / Test
                      │
                      ↓
              CorpusTokenizer
                      │
                      ↓
             GPT-2 Byte BPE
                      │
                      ↓
                EOS Handling
                      │
                      ↓
               Token Streams
                      │
                      ↓
          build_window_indices()
                      │
                      ↓
                GPTDataset
                      │
              ┌───────┴───────┐
              ↓               ↓
     build_input_target()     ...
              │
              ↓
          DataLoader
              │
              ↓
       [Batch, Context]
              │
              ↓
           GPT-2 Model
```

For the current configuration:

```text
Batch size     = 16
Context length = 128
Tokens/step    = 2,048
```

The resulting tensors are:

```text
Input:
[16, 128]

Target:
[16, 128]
```

---

# 30. Key Takeaway

The GPT-2 data pipeline moves from a simple sample-oriented approach toward a reusable language-model pretraining pipeline:

```text
Documents
    ↓
Token Streams
    ↓
Context Windows
    ↓
Input / Target Pairs
    ↓
Batches
```

This separation provides the foundation needed to scale the same architecture later for GPT-3 experiments, where dataset size, token count, context length, batch size, and training steps become much more significant.
