import torch


def compute_policy_loss(
    new_log_probs,
    old_log_probs,
    advantages,
    clip_epsilon=0.2,
    mask=None,
):
    """
    PPO clipped policy loss.

    Args:
        new_log_probs: [batch, response_len]
        old_log_probs: [batch, response_len]
        advantages:    [batch, response_len]
        mask:          [batch, response_len]

    Returns:
        policy_loss
    """

    ratios = torch.exp(
        new_log_probs - old_log_probs
    )

    unclipped = (
        ratios * advantages
    )

    clipped_ratios = torch.clamp(
        ratios,
        1.0 - clip_epsilon,
        1.0 + clip_epsilon,
    )

    clipped = (
        clipped_ratios * advantages
    )

    objective = torch.minimum(
        unclipped,
        clipped,
    )

    if mask is not None:
        objective = objective * mask
        policy_loss = -(
            objective.sum()
            / mask.sum().clamp_min(1.0)
        )
    else:
        policy_loss = -objective.mean()

    return policy_loss


def compute_value_loss(
    values,
    returns,
    mask=None,
):
    """
    Value function loss.
    """

    loss = (
        values - returns
    ).pow(2)

    if mask is not None:
        loss = loss * mask
        return (
            loss.sum()
            / mask.sum().clamp_min(1.0)
        )

    return loss.mean()


def compute_entropy(
    logits,
    mask=None,
):
    """
    Token-level entropy.

    logits:
        [batch, response_len, vocab_size]

    mask:
        [batch, response_len]
    """

    log_probs = torch.log_softmax(
        logits,
        dim=-1,
    )

    probs = torch.exp(log_probs)

    entropy = -(
        probs * log_probs
    ).sum(dim=-1)

    if mask is not None:
        entropy = entropy * mask

        return (
            entropy.sum()
            / mask.sum().clamp_min(1.0)
        )

    return entropy.mean()