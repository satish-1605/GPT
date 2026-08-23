GPT-3 Scaling Strategy
1. Purpose
This document defines the model-scaling strategy for the GPT-3 stage of the project.

The objective is not to reproduce OpenAI's original 175B GPT-3 model. Instead, the goal is to reproduce the core research ideas behind GPT-3 scaling at a smaller and experimentally manageable scale using rented GPUs.

The GPT-3 stage builds directly on the GPT-1 and GPT-2 implementations developed earlier in this project.

The primary research question is:

How does increasing Transformer model capacity affect language-model performance and in-context learning?

The project will therefore train multiple models with progressively increasing parameter counts and compare their:

Parameter count
Model depth
Model width
Attention capacity
FFN capacity
Context length
Training loss
Validation loss
Perplexity
Training throughput
Inference behavior
In-context learning capability
2. Starting Point
The GPT-2 implementation currently serves as the baseline architecture.

The current baseline contains approximately:

GPT-2 Baseline
-------------------------
Parameters       ~6.06M
Vocabulary       5,000
d_model          256
Layers           6
Attention Heads  4
d_ff             1,024
Context Length   128
Head Dimension   64

The GPT-2 baseline has already been trained and evaluated.

The current baseline experiment reached approximately:

Training Steps   4,000
Perplexity       ~171

This model therefore becomes the reference point for the GPT-3 scaling experiments.

3. GPT-3 Scaling Motivation
The original GPT-3 paper investigated whether increasing model capacity results in systematic improvements in language modeling and downstream task performance.

OpenAI trained models ranging from:

125M → 350M → 760M → 1.3B → 2.7B → 6.7B → 13B → 175B

The important idea for this project is not the exact 175B parameter count.

Instead, the important idea is:

Increase model capacity
        ↓
Train models under comparable conditions
        ↓
Measure validation loss
        ↓
Measure downstream behavior
        ↓
Study scaling trends

Our project will apply this methodology at a scale that is feasible with rented GPU resources.

4. Project Scaling Philosophy
The project will follow three principles.

4.1 Multiple Models
Instead of training a single larger model, several models will be trained.

This allows us to study:

Model Size
    ↓
Validation Loss
    ↓
Perplexity
    ↓
Language Generation
    ↓
In-Context Learning

A single model would not allow us to distinguish whether improvements come from scale or from other implementation changes.

4.2 Controlled Architecture
The models should use the same underlying GPT implementation.

The architecture should remain conceptually consistent:

Token Embedding
      ↓
Transformer Blocks
      ↓
Final LayerNorm
      ↓
Language Model Head

The main changes between models will be architectural scale:

d_model
num_layers
num_heads
d_ff
context_length
The implementation itself should not be rewritten for each model.

4.3 Resource-Aware Scaling
The original GPT-3 models were trained using very large distributed GPU infrastructure.

This project has a different objective.

Models will be selected according to:

Research value
      +
Training feasibility
      +
GPU memory
      +
Training time
      +
Rental cost

Therefore, exact parameter counts will be determined after considering the available rented GPU configuration.

5. Target Model Family
The planned model family is:

Model	Approximate Parameters	Purpose
GPT-2 Baseline	~6M	Existing baseline
GPT-3 Mini	~15–25M	Initial scaling experiment
GPT-3 Small	~30–60M	Small-scale scaling
GPT-3 Medium	~80–150M	Medium-scale experiment
GPT-3 Large	~200–350M	Large-scale experiment
GPT-3 XL	~500M–1B	Optional high-scale experiment

These parameter ranges are targets rather than fixed requirements.

The exact configurations will be calculated analytically and validated experimentally.

6. Existing Configurations
Two GPT-3 configurations have already been designed and parameter-count tested.

6.1 GPT-3 Mini
Vocabulary       = 5,000
d_model          = 384
num_heads        = 6
num_layers       = 8
d_ff             = 1,536
context_length   = 256

Approximate parameter count:

~16.22M

Head dimension:

384 / 6 = 64

FFN expansion:

1,536 / 384 = 4

6.2 GPT-3 Small
Vocabulary       = 5,000
d_model          = 512
num_heads        = 8
num_layers       = 10
d_ff             = 2,048
context_length   = 512

Approximate parameter count:

~34.35M

Head dimension:

512 / 8 = 64

FFN expansion:

2,048 / 512 = 4

7. Architecture Invariants
Some architectural relationships will remain approximately constant across the model family.

7.1 Attention Head Dimension
The current configurations use:

d_head = 64

Therefore:

d_model = num_heads × 64

Examples:

256 / 4 = 64
384 / 6 = 64
512 / 8 = 64

Maintaining a consistent head dimension makes the scaling experiments easier to interpret.

7.2 Feed-Forward Expansion
The Transformer FFN follows:

d_ff = 4 × d_model

Examples:

d_model = 256
d_ff    = 1,024

d_model = 384
d_ff    = 1,536

d_model = 512
d_ff    = 2,048

This relationship will be maintained unless a later experiment explicitly investigates alternative FFN scaling.

7.3 Vocabulary
The initial experiments use:

Vocabulary Size = 5,000

This keeps the vocabulary constant so that the primary scaling differences come from the Transformer architecture.

Vocabulary scaling can be investigated separately if required.

8. Scaling Dimensions
Model capacity can be increased through several dimensions.

8.1 Width
Increase:

d_model

Example:

256 → 384 → 512 → 768 → 1024

Increasing width increases the representation capacity of each Transformer layer.

8.2 Depth
Increase:

num_layers

Example:

6 → 8 → 10 → 12 → 16 → 24

Increasing depth increases the number of sequential transformations performed by the model.

8.3 Attention Capacity
Increasing:

num_heads

while maintaining:

d_head = 64

results in:

num_heads = d_model / 64

Examples:

d_model = 256 → 4 heads
d_model = 384 → 6 heads
d_model = 512 → 8 heads

8.4 FFN Capacity
The default relationship is:

d_ff = 4 × d_model

The FFN represents a significant portion of Transformer parameters.

Therefore, increasing d_model automatically increases FFN capacity.

8.5 Context Length
The context window will also be increased as model scale increases where computationally practical.

Current targets:

GPT-2 Baseline → 128
GPT-3 Mini     → 256
GPT-3 Small    → 512

Larger context lengths significantly increase attention computation and memory requirements.

Therefore context length will be treated as a separate experimental variable rather than increased blindly.

9. Parameter Scaling
The exact parameter count will not be selected only from rounded targets such as:

20M
50M
100M
300M

Instead, each configuration will be constructed and then its parameter count will be calculated.

Conceptually:

Target Parameter Range
        ↓
Choose d_model
        ↓
Choose number of layers
        ↓
Choose attention heads
        ↓
Choose d_ff
        ↓
Calculate parameters
        ↓
Instantiate model
        ↓
Verify actual parameters

This ensures that the documented model size matches the actual implementation.

10. Parameter Scaling Baseline
Current model sizes:

Model	Parameters	Relative to GPT-2
GPT-2 Baseline	6.06M	1.00×
GPT-3 Mini	16.22M	~2.68×
GPT-3 Small	34.35M	~5.67×

The relative scaling factor will be used in later experiments to compare model growth.

11. Depth vs Width
One of the research questions in this stage is:

Is increasing depth or width more effective for increasing model capacity?

For example, two models could have similar parameter counts but different architectures:

Model A
More layers
Smaller d_model

Model B
Fewer layers
Larger d_model

Their parameter counts may be similar while their learning behavior differs.

Later experiments may therefore compare:

Depth-heavy model
        vs
Width-heavy model

This will help separate simple parameter scaling from architectural scaling.

12. Attention Head Scaling
Attention heads will initially maintain:

d_head = 64

Therefore:

num_heads = d_model / 64

This creates a consistent relationship between model width and attention capacity.

Later experiments may investigate whether changing the number of heads independently provides measurable benefits.

13. FFN Scaling
The baseline strategy is:

d_ff = 4 × d_model

This follows the standard Transformer/GPT architecture used in the earlier GPT implementations.

Later experiments may investigate:

2× d_model
4× d_model
6× d_model
8× d_model

to understand the effect of FFN capacity independently from attention capacity.

These experiments will only be introduced after establishing the primary scaling baseline.

14. Context Scaling
Context length is computationally expensive because standard self-attention has approximately quadratic complexity with respect to sequence length.

Conceptually:

Attention Cost ∝ context_length²

Therefore:

128 → 256

is approximately a 4× increase in the attention matrix size.

Similarly:

256 → 512

is another approximately 4× increase.

Consequently, context length will be increased carefully based on GPU memory and training economics.

15. GPU Economics
The project will use rented GPUs for larger experiments.

Before training each model, we will estimate:

Model Parameters
        ↓
Parameter Memory
        ↓
Activation Memory
        ↓
Optimizer Memory
        ↓
Total GPU Memory
        ↓
Expected Training Throughput
        ↓
Expected Training Time
        ↓
Estimated Rental Cost

This prevents selecting a model that is theoretically interesting but impractical to train.

