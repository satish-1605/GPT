# GPT-2 — Training

## 1. Overview

The GPT-2 training pipeline optimizes model parameters using autoregressive next-token prediction.

```text
GPT-2 Model
    ↓
Train DataLoader
    ↓
Forward Pass
    ↓
Next-Token Loss
    ↓
Backward Pass
    ↓
Gradient Clipping
    ↓
Learning-Rate Schedule
    ↓
AdamW
    ↓
Parameter Update
```

The training pipeline separates dataset construction, model architecture, optimization, learning-rate scheduling, gradient handling, validation, and checkpointing.

## 2. Training Objective

GPT-2 is trained using causal language modeling.

Given:

```text
x₁, x₂, x₃, ..., xₙ
```

the model learns:

```text
x₁ → x₂
x₂ → x₃
x₃ → x₄
...
xₙ₋₁ → xₙ
```

The objective is to minimize cross-entropy loss:

```text
L = -Σ log P(xₜ | x₁, ..., xₜ₋₁)
```

## 3. Training Components

The training implementation is organized as:

```text
src/training/
├── optimizer.py
├── scheduler.py
├── gradient.py
├── trainer.py
├── validate.py
└── checkpoint.py
```

| Module          | Responsibility             |
| --------------- | -------------------------- |
| `optimizer.py`  | AdamW optimizer            |
| `scheduler.py`  | Learning-rate schedule     |
| `gradient.py`   | Gradient clipping          |
| `trainer.py`    | Training loop              |
| `validate.py`   | Validation loop            |
| `checkpoint.py` | Saving/loading checkpoints |

## 4. AdamW

The model uses AdamW with:

```python
learning_rate = 3e-4
weight_decay = 0.1

beta1 = 0.9
beta2 = 0.95

adam_eps = 1e-8
```

AdamW combines adaptive gradient updates with decoupled weight decay.

## 5. Learning-Rate Schedule

The training pipeline uses:

```python
warmup_steps = 100
min_learning_rate = 3e-5
```

During warmup, the learning rate increases gradually. After warmup, it decreases toward the configured minimum.

## 6. Gradient Clipping

Gradient clipping is applied after backpropagation and before the optimizer update.

```python
max_grad_norm = 1.0
```

The sequence is:

```text
loss.backward()
      ↓
gradient clipping
      ↓
optimizer.step()
```

## 7. Training Step

The current configuration uses:

```text
batch_size = 16
context_length = 128
```

Therefore, one batch contains:

```text
16 × 128 = 2,048
```

token positions.

The training step is:

```text
Batch
 ↓
Move to device
 ↓
Zero gradients
 ↓
Forward pass
 ↓
Calculate loss
 ↓
Backward pass
 ↓
Clip gradients
 ↓
Update learning rate
 ↓
AdamW update
 ↓
global_step += 1
```

In the current implementation, one batch corresponds approximately to one optimizer step.

## 8. Global Step

`global_step` tracks the number of optimizer updates performed.

An epoch represents one pass through the available DataLoader samples, whereas a step represents one optimizer update.

## 9. Epoch vs Step

The training implementation supports both epochs and `max_steps`.

For language-model pretraining, steps are particularly useful because large-scale training is commonly described using optimization steps and tokens processed.

## 10. Training Loop

The high-level process is:

```text
Initialize configuration
        ↓
Create DataLoaders
        ↓
Create GPT-2 model
        ↓
Create AdamW
        ↓
Initialize global_step
        ↓
Train
        ↓
Validate
        ↓
Save best checkpoint
        ↓
Repeat
```

The training entry point is:

```text
src/train.py
```

## 11. Validation During Training

After training for an epoch, the model is evaluated on the validation dataset using:

```python
model.eval()
```

and:

```python
torch.no_grad()
```

No gradients are calculated during validation.

## 12. Best Checkpoint

The best model is selected using validation loss:

```python
if val_loss < best_val_loss:
    save_checkpoint(...)
```

Lower validation loss indicates better next-token prediction on unseen validation data.

## 13. Checkpointing

Checkpoints store:

```text
model_state_dict
optimizer_state_dict
epoch
global_step
best_val_loss
config
```

The checkpoint path is configured through:

```python
checkpoint_path
```

## 14. Resume Training

A checkpoint provides the information required to continue training:

```text
Checkpoint
    ↓
Model state
Optimizer state
Global step
Epoch
Best validation loss
    ↓
Continue training
```

This preserves both model parameters and optimizer state.

## 15. Current Training Configuration

```text
Vocabulary size     = 5,000
Context length      = 128
Batch size          = 16

d_model             = 256
Attention heads     = 4
Transformer layers  = 6
FFN dimension       = 1,024

Learning rate       = 3e-4
Weight decay        = 0.1
β₁                  = 0.9
β₂                  = 0.95

Warmup steps        = 100
Minimum LR          = 3e-5

Gradient norm       = 1.0
```

## 16. Local Training

The GPT-2 model was trained locally during this phase. This validated:

* Model correctness
* Dataset pipeline
* Training loop
* Checkpointing
* Validation
* Inference
* Evaluation

Larger-scale GPU training is reserved for the GPT-3 scaling phase.

## 17. Training Result

The final local training run reached:

```text
Global Steps = 3,466
```

Reported metrics:

```text
Train Loss = 5.6355
Val Loss   = 5.1117
```

The model was subsequently evaluated independently.

## 18. Training Takeaway

The GPT-2 training pipeline now contains the major components required for an autoregressive pretraining workflow:

```text
AdamW
+
Learning-rate scheduling
+
Gradient clipping
+
Validation
+
Checkpointing
+
Global-step tracking
+
Resume support
```