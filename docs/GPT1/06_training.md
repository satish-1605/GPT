## 1. Introduction

After preparing the dataset and implementing the GPT-1 architecture, the next step is training the model. During training, the model learns to predict the next token in a sequence by minimizing the difference between its predictions and the actual target tokens.

The implementation in this project follows the standard autoregressive language modeling objective, where the model predicts the next token given all previous tokens. The target sequence is simply the input sequence shifted by one position, a technique commonly referred to as **teacher forcing**.

The complete training pipeline implemented in this project is illustrated below.

```text
Input Batch
      │
      ▼
Forward Pass
      │
      ▼
Cross Entropy Loss
      │
      ▼
optimizer.zero_grad()
      │
      ▼
Backpropagation
      │
      ▼
AdamW Optimizer
      │
      ▼
Parameter Update
      │
      ▼
Validation
      │
      ▼
Save Best Checkpoint
```

Each stage contributes to improving the model's ability to generate coherent and meaningful text.

---

# 2. Training Configuration

The GPT-1 model was trained using the following configuration.

| Hyperparameter          |              Value |
| ----------------------- | -----------------: |
| Batch Size              |                 16 |
| Learning Rate           |  3 × 10⁻⁴ (0.0003) |
| Optimizer               |              AdamW |
| Loss Function           | Cross Entropy Loss |
| Maximum Sequence Length |                128 |
| Checkpoint File         |    `checkpoint.pt` |

The maximum number of training epochs was configurable through the project configuration. Due to hardware limitations, the experiments presented in this repository were performed for a smaller number of epochs while monitoring the training and validation loss after each epoch.

---

# 3. Forward Pass

During each training iteration, a mini-batch of token sequences is passed through the GPT-1 model.

The model processes the input through the following stages:

* Token Embedding
* Positional Embedding
* Decoder Blocks
* Final Layer Normalization
* Linear Projection

The output is a tensor of **vocabulary logits**, where every position contains a score for each token in the vocabulary.

```text
Input Tokens
        │
        ▼
      GPT-1
        │
        ▼
Vocabulary Logits
```

During training, the target sequence is simply the input sequence shifted one token to the left.

For example,

```text
Input

The cat sat on the

Target

cat sat on the mat
```

This approach enables the model to learn the **next-token prediction** objective.

---

# 4. Cross Entropy Loss

GPT-1 is trained as a **next-token prediction** model.

For every position in the sequence, the model predicts a probability distribution over the entire vocabulary. These predictions are compared with the corresponding target tokens using **Cross Entropy Loss**.

A lower loss indicates that the predicted probability distribution is closer to the true target distribution.

For example,

```text
Input

The cat sat on the

Target

mat
```

If the model assigns a high probability to the correct token (**"mat"**), the loss is small. Otherwise, the loss increases.

Cross Entropy Loss is well suited for language modeling because every vocabulary token represents a separate prediction class.

---

# 5. Backpropagation

After computing the loss, the optimizer first clears the gradients accumulated from the previous iteration.

```python
optimizer.zero_grad()
```

Next, gradients are computed using backpropagation.

```python
loss.backward()
```

Finally, the optimizer updates all trainable parameters.

```python
optimizer.step()
```

This process is repeated for every mini-batch during training.

---

# 6. AdamW Optimizer

The model parameters are updated using the **AdamW** optimizer.

AdamW extends the Adam optimizer by decoupling weight decay from the gradient update, making it the preferred optimizer for modern Transformer-based language models.

During every training iteration, the optimizer performs the following steps:

1. Clear gradients from the previous iteration.
2. Compute gradients using backpropagation.
3. Update all trainable parameters.
4. Prepare for the next training iteration.

The learning rate used in this project is

```text
0.0003
```

---

# 7. Validation

After each training epoch, the model is evaluated on the validation dataset.

During validation,

* The model is switched to evaluation mode using `model.eval()`.
* Gradient computation is disabled using `torch.no_grad()`.

Unlike training, no parameter updates are performed during validation.

Validation helps to:

* Measure the model's generalization performance.
* Monitor training progress.
* Detect overfitting.
* Select the best-performing checkpoint.

The average validation loss is recorded after every epoch.

---

# 8. Checkpoint Saving

To preserve the best-performing model, checkpoints are automatically saved during training.

Whenever the validation loss improves, the current model state is stored in

```text
checkpoint.pt
```

The checkpoint saved in this project contains:

* Model state dictionary
* Optimizer state dictionary
* Current epoch
* Training loss
* Validation loss

Saving checkpoints provides several advantages:

* Resume interrupted training.
* Prevent loss of training progress.
* Restore the best-performing model.
* Perform inference and evaluation without retraining.

---

# 9. Training Workflow

The complete training procedure implemented in this project is summarized below.

```text
Load Training Batch
        │
        ▼
Forward Pass
        │
        ▼
Compute Cross Entropy Loss
        │
        ▼
optimizer.zero_grad()
        │
        ▼
Backpropagation
        │
        ▼
AdamW Optimizer Update
        │
        ▼
Next Batch
        │
        ▼
Validation
        │
        ▼
Save Best Checkpoint
```

This workflow is repeated for every epoch until training is complete.

---

# 10. Training and Validation Monitoring

Throughout training, both the training loss and validation loss are monitored after each epoch.

* **Training Loss** measures how well the model fits the training dataset.
* **Validation Loss** measures how well the model generalizes to unseen data.

The checkpoint with the **lowest validation loss** is selected as the final model, helping to reduce overfitting.

---

# 11. Best Validation Performance

The best-performing checkpoint is selected based on the minimum validation loss observed during training.

| Metric               |                                                Value |
| -------------------- | ---------------------------------------------------: |
| Best Validation Loss |          **≈ 5.4** *(Replace with your exact value)* |
| Best Epoch           | **4** *(Replace with your exact epoch if different)* |

---

# 12. Summary

The GPT-1 model was trained using the **AdamW** optimizer and **Cross Entropy Loss** to learn the next-token prediction objective. During each iteration, input sequences were processed through the GPT-1 model, the prediction error was computed, and gradients were propagated backward to update the model parameters.

After every epoch, the model was evaluated on a validation dataset to monitor generalization performance. The checkpoint corresponding to the lowest validation loss was automatically saved, allowing the trained model to be reused for inference and evaluation without retraining.

Although the model was trained on a subset of the TinyStories dataset due to hardware limitations, the complete training pipeline closely follows the methodology used in modern decoder-only Transformer language models.
