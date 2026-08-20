from src.training.scheduler import get_learning_rate

def test_learning_rate_warmup():

    max_lr = 3e-4
    warmup_steps = 100

    lr_0 = get_learning_rate(
        step=0,
        max_steps=1000,
        learning_rate=max_lr,
        warmup_steps=warmup_steps,
        min_learning_rate=3e-5,
    )

    lr_50 = get_learning_rate(
        step=50,
        max_steps=1000,
        learning_rate=max_lr,
        warmup_steps=warmup_steps,
        min_learning_rate=3e-5,
    )

    lr_100 = get_learning_rate(
        step=100,
        max_steps=1000,
        learning_rate=max_lr,
        warmup_steps=warmup_steps,
        min_learning_rate=3e-5,
    )

    assert lr_0 == 0.0
    assert lr_50 == 1.5e-4
    assert lr_100 == max_lr

if __name__ == "__main__":
    test_learning_rate_warmup()