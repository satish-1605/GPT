## 1. Overview

GPT-3 (Generative Pre-trained Transformer 3) was introduced in:

> **Language Models are Few-Shot Learners**  
> Brown et al., 2020

GPT-3 is a large-scale autoregressive Transformer language model that investigates how increasing model capacity, training data, and compute affects language modeling and downstream task performance.

The major conceptual contribution of GPT-3 is not a fundamentally new Transformer architecture. Instead, GPT-3 demonstrates that **scaling an autoregressive language model can substantially improve its ability to perform tasks directly from natural-language context without gradient-based fine-tuning**.

This capability is commonly referred to as:

> **In-context learning**

GPT-3 evaluates this capability through:

- Zero-shot learning
- One-shot learning
- Few-shot learning

The GPT-3 paper also studies:

- Scaling behavior
- Dataset composition
- Training compute
- Data contamination
- Model-size effects
- Limitations and broader impacts

---

# 2. GPT-3 Research Question

The central question of GPT-3 is:

> How does the ability of a language model to perform tasks from context change as the model is scaled?

Previous NLP systems typically followed:

```text
Large-scale pretraining
        ↓
Task-specific dataset
        ↓
Fine-tuning
        ↓
Task-specific model

GPT-3 investigates an alternative:
Large-scale pretraining
        ↓
General language model
        ↓
Task description + examples
        ↓
Prediction

No task-specific gradient updates are performed during inference.

3. Motivation
Traditional pretraining + fine-tuning approaches have several limitations.

3.1 Task-Specific Datasets
Fine-tuning generally requires thousands to hundreds of thousands of labeled examples for every task.

This makes it expensive to adapt a model to many different tasks.

3.2 Narrow Task Distributions
A model may become highly specialized to the distribution of a particular fine-tuning dataset.

This can result in poor out-of-distribution generalization.

3.3 Human Learning Efficiency
Humans can often learn a new task from:

A natural-language instruction
One example
A few examples
GPT-3 investigates whether sufficiently large language models can demonstrate similar behavior.

4. GPT-3 Scaling Hypothesis
GPT-3 is based on the observation that language-model performance tends to improve predictably with model scale.

Three major scaling dimensions are considered:

Model size
Dataset / training tokens
Compute
Increasing these dimensions can lead to improved:

Validation loss
Language modeling performance
Zero-shot performance
One-shot performance
Few-shot performance
In-context learning ability
The GPT-3 experiments therefore use multiple model sizes rather than evaluating only one large model.

5. GPT-3 Model Family
GPT-3 was trained in eight different sizes.

Model	Parameters	Layers	d_model	Heads	Batch Size
GPT-3 Small	125M	12	768	12	0.5M
GPT-3 Medium	350M	24	1024	16	0.5M
GPT-3 Large	760M	24	1536	16	0.5M
GPT-3 XL	1.3B	24	2048	24	1M
GPT-3 2.7B	2.7B	32	2560	32	1M
GPT-3 6.7B	6.7B	32	4096	32	2M
GPT-3 13B	13B	40	5140	40	2M
GPT-3	175B	96	12288	96	3.2M

All models use:

Context length = 2048 tokens
d_ff = 4 × d_model
The largest model contains approximately 175 billion parameters.

6. GPT-3 Architecture
GPT-3 uses an architecture based on GPT-2.

Important architectural characteristics include:

Autoregressive Transformer
Decoder-only architecture
Pre-normalization
Modified parameter initialization
Byte-level BPE tokenization
2048-token context window
Dense and locally banded sparse attention patterns
The core architecture remains fundamentally a Transformer language model.

The major innovation of GPT-3 is therefore primarily:

Scaling + training methodology + in-context learning

rather than introducing a completely new architecture.

7. Model Scaling
GPT-3 investigates models ranging from:

125M → 175B parameters

This represents approximately three orders of magnitude of model capacity.

The purpose of training multiple models is to investigate whether performance follows a smooth relationship with model size.

Conceptually:

Model size
     ↓
Representation capacity
     ↓
Language modeling ability
     ↓
Task performance
     ↓
In-context learning ability

For our implementation, we cannot reproduce the 175B model.

Instead, we will investigate the same scaling principle at a smaller scale.

Our experimental model family may include models such as:

GPT-2 baseline      ~6M
GPT-3 Mini          ~20M
GPT-3 Small         ~50M
GPT-3 Medium        ~100M
GPT-3 Large         ~300M

The exact configurations will be determined during the architecture and scaling stages.

8. Dataset Scaling
GPT-3 uses a large mixture of internet and curated datasets.

The main training datasets are:

Dataset	Tokens	Training Weight
Filtered Common Crawl	410B	60%
WebText2	19B	22%
Books1	12B	8%
Books2	55B	8%
Wikipedia	3B	3%

The important point is that:

Dataset size and training-token count are not the same thing.

The datasets are sampled using different weights.

Higher-quality datasets are intentionally sampled more frequently.

9. Common Crawl Processing
Raw Common Crawl contains a large amount of noisy content.

GPT-3 therefore applies several preprocessing strategies.

9.1 Quality Filtering
Common Crawl documents are filtered based on similarity to high-quality reference datasets.

9.2 Fuzzy Deduplication
Documents are deduplicated to reduce repeated content.

Deduplication is performed:

Within datasets
Across datasets
9.3 Curated Datasets
High-quality datasets are added to improve:

Quality
Diversity
Language coverage
Knowledge representation
10. Training Token Budget
GPT-3 models were trained for:

300 billion tokens

However, because different datasets have different sampling weights, some datasets are effectively seen multiple times while others are not completely consumed.

For example:

Dataset	Approximate Epochs
Common Crawl	< 1 epoch
Books2	< 1 epoch
WebText2	~3 epochs
Books1	~2 epochs
Wikipedia	~3.4 epochs

This demonstrates an important training principle:

Training should not necessarily sample every dataset strictly according to its raw size.

Dataset quality can influence sampling frequency.

11. Compute Scaling
GPT-3 requires enormous computational resources.

The original models were trained on NVIDIA V100 GPUs using large-scale distributed training.

The largest models required:

Model parallelism
GPU memory partitioning
Distributed training
Parallel matrix multiplication
Pipeline/layer parallelism
The important research principle is:

Model size
      ×
Training tokens
      ×
Training compute

Our implementation will investigate the same relationship at a much smaller scale using rented GPUs.

We will track:

GPU type
GPU memory
Number of GPUs
Training time
Tokens/second
Training steps
Training tokens
GPU-hours
Estimated compute cost
12. Batch Size Scaling
GPT-3 follows the observation that larger models can generally benefit from larger batch sizes.

The paper also uses gradient noise scale measurements to guide batch-size selection.

As model size increases:

Model size ↑
     ↓
Batch size can ↑

while:

Model size ↑
     ↓
Learning rate generally ↓

This is important for our scaling experiments.

13. Learning Rate Scaling
GPT-3 uses smaller learning rates for larger models.

Model	Learning Rate
125M	6.0 × 10⁻⁴
350M	3.0 × 10⁻⁴
760M	2.5 × 10⁻⁴
1.3B	2.0 × 10⁻⁴
2.7B	1.6 × 10⁻⁴
6.7B	1.2 × 10⁻⁴
13B	1.0 × 10⁻⁴
175B	0.6 × 10⁻⁴

Therefore, our implementation should not assume that a single learning rate is optimal for every model size.

14. Zero-Shot Learning
Zero-shot learning means that no task-specific examples are provided.

The model receives a natural-language description of the task.

Example:

Translate English to French.

English: Hello
French:

The model generates:

Bonjour

There are:

No demonstrations
No gradient updates
No fine-tuning
Only the pretrained model is used.

15. One-Shot Learning
One-shot learning provides exactly one demonstration.

Example:

Translate English to French.

English: Good morning.
French: Bonjour.

English: How are you?
French:

The model uses the single demonstration to infer the desired behavior.

Again:

No weight update
No gradient computation
No fine-tuning
16. Few-Shot Learning
Few-shot learning provides multiple demonstrations.

Example:

English: Hello
French: Bonjour

English: Good night
French: Bonne nuit

English: Thank you
French: Merci

English: How are you?
French:

The model infers the task from the examples in the context.

GPT-3 typically uses approximately:

10–100 examples

depending on the task and context-window limitations.

17. In-Context Learning
In-context learning is one of the central ideas of GPT-3.

The model receives:

Task description
      +
Demonstrations
      +
New input
      ↓
Transformer
      ↓
Prediction

The model parameters remain unchanged.

There is no:

Backpropagation
Optimizer step
Weight update
Fine-tuning
The adaptation occurs through the input context.

18. Fine-Tuning vs. In-Context Learning
Property	Fine-Tuning	In-Context Learning
Task-specific dataset	Required	Optional
Examples	Thousands+	0–100 typically
Gradient updates	Yes	No
Weight updates	Yes	No
Training per task	Yes	No
Inference adaptation	Parameter-based	Context-based
Zero-shot possible	Not traditionally	Yes
One-shot possible	Not traditionally	Yes
Few-shot possible	Not traditionally	Yes

GPT-3 focuses primarily on:

Zero-shot
One-shot
Few-shot
rather than traditional fine-tuning.

19. Context Window
GPT-3 uses:

n_ctx = 2048 tokens

This context window limits:

Number of demonstrations
Prompt length
Input length
Generated sequence context
Therefore:

Context length
      ↓
Number of examples that fit
      ↓
Few-shot capability

This makes context length an important architectural parameter.

20. GPT-3 Evaluation
GPT-3 evaluates a broad range of tasks.

Major Categories
Language Modeling
LAMBADA
Penn Treebank
WikiText
Question Answering
TriviaQA
Natural Questions
WebQuestions
CoQA
Translation
English → French
English → German
English → Romanian
Reasoning
Winograd
Winogrande
ARC
OpenBookQA
PIQA
Reading Comprehension
RACE
QuAC
Natural Language Inference
ANLI
Other NLI benchmarks
General NLP
SuperGLUE
Synthetic Tasks
GPT-3 is also evaluated on tasks designed to test:

Pattern recognition
Arithmetic
Word manipulation
Novel word usage
On-the-fly task adaptation
21. Evaluation Settings
For each task, GPT-3 evaluates:

Zero-shot
One-shot
Few-shot
This allows researchers to measure how performance changes as demonstrations increase.

Conceptually:

0 examples
     ↓
1 example
     ↓
10 examples
     ↓
50 examples
     ↓
100 examples

The resulting performance curve provides information about the model's in-context learning ability.

22. Model Scaling and In-Context Learning
One of the most important observations from GPT-3 is:

Larger models become increasingly effective at using information provided in context.

Conceptually:

Model size ↑
      ↓
In-context learning ability ↑
      ↓
Few-shot performance ↑

The gap between:

Zero-shot
One-shot
Few-shot
can become more pronounced as model size increases.

This suggests that larger models are better at exploiting demonstrations.

23. Data Contamination
Large web-scale datasets create an important evaluation problem:

The benchmark being evaluated may already exist in the training corpus.

For example:

Benchmark test example
        ↓
appears on the internet
        ↓
Common Crawl
        ↓
training dataset
        ↓
model training
        ↓
benchmark evaluation

The model may therefore memorize the benchmark.

This can artificially inflate evaluation results.

24. Contamination Detection
GPT-3 investigates overlap between training data and benchmark datasets.

Important techniques include:

Exact matching
Fuzzy matching
Document-level filtering
Benchmark overlap analysis
The goal is to distinguish:

Generalization

from:

Memorization

This will also become part of our research implementation.

25. GPT-3 Limitations
GPT-3 does not solve all NLP problems.

The paper identifies weaknesses in areas such as:

Natural language inference
Reading comprehension
Reasoning
Some benchmark tasks
Tasks requiring precise multi-step reasoning
Performance also varies considerably across tasks.

Large scale does not guarantee perfect reasoning.

26. GPT-3 Broader Implications
The paper discusses several broader concerns.

Misuse
Large language models can potentially be used for:

Spam
Misinformation
Automated content generation
Manipulation
Bias
Large internet datasets can contain:

Social biases
Stereotypes
Toxic language
Unequal representation
These can be learned by the model.

Energy Consumption
Large-scale model training requires substantial computational resources and therefore significant energy.