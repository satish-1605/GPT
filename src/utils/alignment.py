def build_input_target(
    token_stream: list[int],
    start: int,
    context_length: int
) -> tuple[list[int], list[int]]:

    input_ids = token_stream[
        start : start + context_length
    ]

    target_ids = token_stream[
        start + 1 : start + context_length + 1
    ]

    return input_ids, target_ids