from src.losses.loss import calculate_loss


def train_one_epoch(loader, model, optimizer, device):
    total_loss = 0

    model.train()

    for input_ids, target_ids in loader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        logits = model(input_ids)
        loss = calculate_loss(logits, target_ids)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()
    avg_loss = total_loss/len(loader)
    return avg_loss




         
    
