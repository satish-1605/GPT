import torch.nn as nn

def calculate_loss(
        logits, target_ids
    ):
    """
        logits shape : (B, S, V)
            where B : Batch Size
                  S: sequence length
                  V: vocab size

        target_ids shape: (B, S)
                where B : Batch Size
                      S: sequence length

        nn.CrossEntropyLoss expects :
                Predictions : [N, C]
                Targets     : [N]
    """
    logits = logits.view(-1, logits.size(-1))
    target_ids = target_ids.view(-1)

    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, target_ids)
    return loss