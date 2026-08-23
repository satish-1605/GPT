import math

def get_learning_rate(
        step:int,
        max_steps :int,
        learning_rate:float,
        warmup_steps:int,
        min_learning_rate:float,
        )->float:
    if step < warmup_steps:
        return learning_rate * (step / warmup_steps)

    if step >= max_steps:
        return min_learning_rate

    decay_steps = max_steps - warmup_steps

    decay_progress = (step - warmup_steps) / decay_steps

    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * decay_progress))

    return (
        min_learning_rate
        + cosine_decay
        * (
            learning_rate
            - min_learning_rate
        )
    )

