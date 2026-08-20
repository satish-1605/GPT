from src.utils.config import GPTConfig

def test_training_config():

    config = GPTConfig()

    assert config.learning_rate == 3e-4
    assert config.weight_decay == 0.1

    assert config.beta1 == 0.9
    assert config.beta2 == 0.95
    assert config.adam_eps == 1e-8

    assert config.max_grad_norm == 1.0
    assert config.warmup_steps == 100

    assert config.epochs == 1
    assert config.max_steps is None

if __name__ == "__main__":
    test_training_config()