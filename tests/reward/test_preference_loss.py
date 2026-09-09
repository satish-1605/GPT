import torch

from src.alignment.reward.reward_loss import preference_loss


def main():

    # Case 1:
    # Correct ordering
    # Chosen reward > Rejected reward
    chosen_rewards_good = torch.tensor([2.0])
    rejected_rewards_good = torch.tensor([0.5])

    good_loss = preference_loss(
        chosen_rewards_good,
        rejected_rewards_good,
    )

    print(f"Good ordering loss: {good_loss.item():.4f}")


    # Case 2:
    # Wrong ordering
    # Chosen reward < Rejected reward
    chosen_rewards_bad = torch.tensor([0.5])
    rejected_rewards_bad = torch.tensor([2.0])

    bad_loss = preference_loss(
        chosen_rewards_bad,
        rejected_rewards_bad,
    )

    print(f"Bad ordering loss: {bad_loss.item():.4f}")


    # Validation
    assert good_loss < bad_loss, (
        "Expected lower loss when chosen reward "
        "is greater than rejected reward."
    )

    print("\nPreference loss test passed successfully.")


if __name__ == "__main__":
    main()