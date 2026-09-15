import torch


class RewardKL:

    def __init__(
        self,
        reward_model,
        kl_coef=0.1,
    ):
        self.reward_model = reward_model
        self.kl_coef = kl_coef

    @torch.no_grad()
    def compute(
        self,
        input_ids,
        old_log_probs,
        ref_log_probs,
        attention_mask,
        response_mask,
    ):

        # ==================================================
        # 1. Reward Model score
        # ==================================================

        rm_reward = self.reward_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        rm_reward = rm_reward.squeeze(-1)

        # ==================================================
        # 2. Token-level KL approximation
        # ==================================================

        kl = (
            old_log_probs
            - ref_log_probs
        )

        # ==================================================
        # 3. KL penalty
        # ==================================================

        kl_penalty = (
            self.kl_coef * kl
        )

        # ==================================================
        # 4. KL reward
        # ==================================================

        rewards = -kl_penalty

        # Ignore invalid/padded response positions
        rewards = rewards * response_mask

        # ==================================================
        # 5. Add RM reward to final response token
        # ==================================================

        final_token_index = (
            response_mask.sum(dim=1).long() - 1
        )

        batch_indices = torch.arange(
            rewards.shape[0],
            device=rewards.device,
        )

        rewards[
            batch_indices,
            final_token_index,
        ] += rm_reward

        return {
            "rm_reward": rm_reward,
            "kl": kl,
            "kl_penalty": kl_penalty,
            "rewards": rewards,
        }