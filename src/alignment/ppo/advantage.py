import torch


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> torch.Tensor:

    """
    Compute Generalized Advantage Estimation (GAE).

    Args:
        rewards: [batch, response_len]
        values:  [batch, response_len]

    Returns:
        advantages: [batch, response_len]
        returns:    [batch, response_len]
    """

    batch_size, sequence_length = rewards.shape

    advantages = torch.zeros_like(rewards)

    A_next = torch.zeros(
        batch_size,
        device=rewards.device,
        dtype=rewards.dtype,
    )

    for t in reversed(range(sequence_length)):

        if t == sequence_length - 1:
            td_error = rewards[:, t] - values[:, t]
        else:
            td_error = (
                rewards[:, t]
                + gamma * values[:, t + 1]
                - values[:, t]
            )

        A_next = td_error + gamma * lam * A_next

        advantages[:, t] = A_next

    returns = advantages + values

    return advantages, returns