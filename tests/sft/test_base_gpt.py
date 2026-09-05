import torch
from src.models.gpt import GPT
from src.utils.config import GPTConfig
config = GPTConfig()
from src.datasets.data_pipeline import get_train_val_test_loaders
from src.losses.loss import calculate_loss
train_loader, _, _ = get_train_val_test_loaders(config)

device = "cpu"

model = GPT(config).to(device)
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

x, y = train_loader.dataset[0]
x = x.unsqueeze(0).to(device)
y = y.unsqueeze(0).to(device)

for step in range(300):
    optimizer.zero_grad()

    logits = model(x)
    loss = calculate_loss(logits, y)

    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step}: Loss {loss.item():.4f}")