## 1. Overview

Phase 1 implements a decoder-only GPT language model from scratch.

The implementation progresses through:

1. GPT-1
2. GPT-2 architectural improvements
3. GPT-3-inspired scaling
4. Final Base GPT

The final milestone is a approximately **300M parameter Base GPT model** trained on approximately **100k documents** using a **32K GPT-2 Byte-Level BPE tokenizer**.

The purpose of this phase is to build and understand the complete pretraining pipeline before moving to instruction tuning and alignment.

---

# 2. Phase 1 Roadmap

```text
Phase 1
│
├── GPT-1
│
├── GPT-2 Improvements
│
├── GPT-3 Scaling
│
└── Final Base GPT

# 2. Final Base GPT Specifications

Component	Configuration
Model Type	Decoder-only Transformer
Parameters	~300M
Vocabulary Size	32,000
Context Length	1,024
Hidden Dimension	1,024
Transformer Layers	24
Attention Heads	16
Head Dimension	64
FFN Dimension	4,096
Dropout	0.1
Normalization	LayerNorm
Activation	GELU
Attention	Causal Self-Attention
Embeddings	Token + Positional
Optimizer	AdamW

4. Dataset
The final model was trained using a FineWeb-style text corpus containing approximately: 100,000

Raw Dataset
     │
     ▼
Cleaning / Preprocessing
     │
     ▼
Train / Validation / Test Split
     │
     ▼
GPT-2 Byte-Level BPE Tokenization
     │
     ▼
Token Streams
     │
     ▼
Fixed-Length Training Sequences
     │
     ▼
Pre-tokenized .pt Files

final dataset artifacts:
    train.pt
    val.pt
    test.pt

The tokenizer was trained using the training portion of the corpus.

5. Tokenizer
    Configuration:

    Tokenizer       : GPT-2 Byte-Level BPE
    Vocabulary      : 32,000
    Special Token   : <|endoftext|>
    Byte Level      : Yes

    Tokenizer artifacts:
    artifacts/
    └── tokenizer/
        ├── vocab.json
        ├── merges.txt
        └── config.json
    
    The tokenizer was trained on the final training corpus.

The tokenizer uses byte-level encoding, allowing it to represent arbitrary UTF-8 text without requiring an unknown-token mechanism.

6. Data Pipeline

The final training pipeline uses pre-tokenized datasets to avoid repeatedly performing expensive tokenization during model training.

Documents
    │
    ▼
Preprocessing
    │
    ▼
Train / Validation / Test
    │
    ▼
Tokenizer
    │
    ▼
Token IDs
    │
    ▼
Token Streams
    │
    ▼
Context Windows
    │
    ▼
PyTorch Dataset
    │
    ▼
DataLoader

Each training example contains:

Input:
x = tokens[0 : context_length]

Target:
y = tokens[1 : context_length + 1]

7. Model Architecture

The final model follows the GPT decoder-only Transformer architecture.

Input Token IDs
       │
       ▼
Token Embeddings
       +
Positional Embeddings
       │
       ▼
┌───────────────────────┐
│ Transformer Block     │
│                       │
│ LayerNorm             │
│      ↓                │
│ Causal Self-Attention │
│      ↓                │
│ Residual Connection   │
│      ↓                │
│ LayerNorm             │
│      ↓                │
│ Feed Forward Network  │
│      ↓                │
│ Residual Connection   │
└───────────────────────┘
       │
       │ × 24
       ▼
Final LayerNorm
       │
       ▼
Language Model Head
       │
       ▼
Vocabulary Logits

The attention mechanism is causal, ensuring that each token can only attend to previous tokens.

8. GPT-3 Scaling
The project explored multiple model sizes before selecting the final Base GPT configuration.
| Model          | Parameters |
| -------------- | ---------: |
| GPT-2 Baseline |        ~6M |
| GPT-3 Mini     |       ~16M |
| GPT-3 Small    |       ~34M |
| Final Base GPT |      ~300M |

The final model was selected as the largest practical configuration for the available GPU hardware.

9. Training Configuration

Final training configuration:

Dataset              : ~100K documents
Vocabulary           : 32,000
Context Length       : 1024

Model Dimension      : 1024
Layers               : 24
Attention Heads      : 16
FFN Dimension        : 4096

Batch Size           : 2

Learning Rate        : 3e-4
Weight Decay         : 0.1
Adam Beta1           : 0.9
Adam Beta2           : 0.95
Adam Epsilon         : 1e-8
Gradient Clipping    : 1.0
Maximum Steps        : 30,000

10. Training Hardware

Training was performed using:
GPU     : NVIDIA RTX 4090
VRAM    : 24 GB
The final 30K-step training run took approximately: 140 min
The RTX 4090 was sufficient to train the final ~300M parameter model using the selected batch size and context length.

11. Training Results

Final training result:
Global Steps : 30,000

Train Loss   : 5.0688
Val Loss     : 5.0351
Perplexity : 153.7081

The model showed continued improvement compared with the earlier 10K-step experiment.
| Metric          | 10K Steps |    30K Steps |
| --------------- | --------: | -----------: |
| Validation Loss |    5.8086 |   **5.0351** |
| Perplexity      |  333.1689 | **153.7081** |


12. Inference
The trained model was evaluated using autoregressive generation.
Prompt
  │
  ▼
Tokenizer
  │
  ▼
Token IDs
  │
  ▼
GPT Model
  │
  ▼
Next-token probabilities
  │
  ▼
Sampling
  │
  ▼
Generated Token
  │
  └──────► Repeat

The following decoding strategies were implemented:

Greedy decoding
Temperature sampling
Top-k sampling
Top-p sampling

13. Qualitative Evaluation

Example prompts included:
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

The model demonstrated:

    English text generation
    Local grammatical structure
    Basic narrative continuation
    Ability to generate multi-sentence text
    Some contextual continuation

However, repetitive patterns were frequently observed.

For example, greedy decoding produced repeated phrases such as:
The city is located in the city...
    The first time...
    The first time...
    The first time...

This indicates that the model has learned useful language patterns but remains far from the quality of a production-scale language model.

14. Sampling Evaluation
Greedy Sampling:
    Produces deterministic output but showed significant repetition.

Temperature Sampling:
    Introduced additional diversity but sometimes produced incoherent or repetitive text.

Top-k Sampling:
    Generally produced more diverse and locally coherent generations.

Top-p Sampling:
    Produced highly diverse generations but occasionally generated noisy or unrelated text.

    Greedy
    ↓
    High repetition

    Temperature
    ↓
    More diversity

    Top-k
    ↓
    Better balance

    Top-p
    ↓
    Highest diversity but more noise

15. In-Context Learning

The Base GPT was evaluated for basic in-context learning.
Experiments included:
    0-shot
    1-shot
    3-shot
    5-shot

Two categories were tested.
    1. Synthetic Pattern Completion
    Eg.
    The cat is an animal.
    The dog is an animal.
    The rose is a ...

    The model was asked to infer the continuation.
    The model did not reliably produce the expected answer.

    2. Natural Text Continuation
    eg. 
    The little boy went to the park.
    He saw a dog.
    The dog was playing with a ball.

    The model was then given a new narrative pattern and asked to continue it.
    The model generated plausible text in some cases, but increasing the number of demonstrations did not produce a consistent improvement.

16. ICL Results

| Shots  | Result                    |
| ------ | ------------------------- |
| 0-shot | Weak                      |
| 1-shot | Weak / partially coherent |
| 3-shot | Weak / partially coherent |
| 5-shot | Weak / partially coherent |

Conclusion

The model did not demonstrate strong or consistent in-context learning.

This result is expected given:

~300M parameters
Limited training corpus
30K training steps
Training from scratch
No instruction tuning
No supervised fine-tuning
No alignment training

The ICL implementation itself was successfully completed and tested.


18. Limitations

The final Base GPT has several limitations.

Dataset

The training corpus is much smaller than datasets used for modern large language models.

Model Size

The model contains approximately 300M parameters, which is significantly smaller than modern foundation models.

Training Budget

The model was trained for 30K optimization steps.

Generation Quality

Generation can become repetitive and incoherent, particularly with greedy decoding.

ICL

Strong in-context learning was not observed.

Alignment

The model is a base language model and has not undergone:

Instruction tuning
RLHF
DPO
Preference optimization

Therefore, it is not intended to behave as a conversational assistant yet.