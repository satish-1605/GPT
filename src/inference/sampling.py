import torch

def greedy_sampling(logits):
    """
    Select the next token using greedy decoding.

    Args:
        logits (torch.Tensor):
            Shape: (batch_size, seq_len, vocab_size)

    Returns:
        torch.Tensor:
            Shape: (batch_size,)
    """
    last_token_logits = logits[:, -1, :]
    next_token_id  = torch.argmax(last_token_logits, dim=-1)
    return next_token_id

def temperature_sampling(logits, T):
    """
    Sample the next token using temperature sampling.
    """

    if T <= 0:
        raise ValueError("Temperature must be greater than 0.")

    last_token_logits = logits[:, -1, :]

    last_token_logits = last_token_logits / T
    probabilities = torch.softmax(last_token_logits, dim=-1)

    sample_token = torch.multinomial(probabilities, num_samples=1).squeeze(-1)
    return sample_token

def top_k_sampling(logits , k):
    if k <= 0:
            raise ValueError("k must be greater than 0.")

    next_token_logits = logits[:, -1, :]

    topk_logits, topk_indices = torch.topk(next_token_logits, k,dim=-1)

    probs = torch.softmax(topk_logits, dim=-1)
    

    sampled_idx = torch.multinomial(
        probs,
        num_samples=1,
    )

    sample_token = torch.gather(
        topk_indices,
        dim=1,
        index=sampled_idx,
    ).squeeze(-1)
    return sample_token


def top_p_sampling(logits, p):
    if not (0 < p <= 1):
        raise ValueError("p must be in (0, 1].")

    next_token_logits = logits[:, -1, :]

    probs = torch.softmax(next_token_logits, dim=-1)

    sorted_probs, sorted_indices = torch.sort(
                                    probs,
                                    descending=True,
                                    dim=-1,
                                )

    cumulative_probs = torch.cumsum(
                            sorted_probs,
                            dim=-1,
                        )

    mask = cumulative_probs > p

    mask[..., 1:] = mask[..., :-1].clone()

    mask[..., 0] = False
    sorted_probs[mask] = 0

    sorted_probs = sorted_probs / sorted_probs.sum(
        dim=-1,
        keepdim=True,
    )

    sampled_idx = torch.multinomial(
        sorted_probs,
        num_samples=1,
    )
    sample_token = torch.gather(
        sorted_indices,
        dim=1,
        index=sampled_idx,
    ).squeeze(-1)

    return sample_token



    
    