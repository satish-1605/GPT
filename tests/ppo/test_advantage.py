import torch
from src.alignment.ppo.advantage import compute_gae

def test_compute_gae_manual_example():
    rewards = torch.tensor(
        [[0.23, 0.45, 0.65, 0.87]],
        dtype=torch.float32,
    )

    values = torch.tensor(
        [[0.50, 0.34, 0.55, 0.90]],
        dtype=torch.float32,
    )

    advantages = compute_gae(rewards, values)

    expected = torch.tensor(
        [[1.5338, 1.5600, 0.9628, -0.0300]],
        dtype=torch.float32,
    )

    assert advantages.shape == rewards.shape

    assert torch.allclose(
        advantages,
        expected,
        atol=1e-3,
    )

    def test_compute_gae_supports_batch():

        rewards = torch.tensor(
            [
                [0.23, 0.45, 0.65, 0.87],
                [0.10, 0.20, 0.30, 0.40],
            ],
            dtype=torch.float32,
        )

        values = torch.tensor(
            [
                [0.50, 0.34, 0.55, 0.90],
                [0.20, 0.30, 0.40, 0.50],
            ],
            dtype=torch.float32,
        )

        advantages = compute_gae(rewards, values)

        assert advantages.shape == (2, 4)

        assert torch.isfinite(advantages).all()