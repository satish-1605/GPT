def build_window_indices(
    token_stream: list[int],
    context_length: int,
    stride: int
) -> list[int]:

    if context_length <= 0:
        raise ValueError(
            "context_length must be greater than 0"
        )

    if stride <= 0:
        raise ValueError(
            "stride must be greater than 0"
        )

    window_indices = []

    for start in range(
        0,
        len(token_stream) - context_length,
        stride
    ):
        window_indices.append(start)

    return window_indices