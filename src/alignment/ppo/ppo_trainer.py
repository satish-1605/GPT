import torch

from transformers import GPT2LMHeadModel, GPT2Tokenizer

from src.alignment.ppo.value_model import ValueModel
from src.alignment.ppo.rollout import Rollout
from src.alignment.ppo.reward_kl import RewardKL

from src.alignment.reward.reward_model import GPT2RewardModel
from src.alignment.ppo.ppo_losses import compute_policy_loss, compute_value_loss, compute_entropy
from src.alignment.ppo.advantage import compute_gae

class PPOTrainer:
    def __init__(self, model_name, sft_checkpoint_path, reward_checkpoint_path, device="cuda",
                 kl_coef=0.1, value_coef=0.5, entropy_coef=0.01,
                 clip_epsilon=0.2, gamma=0.99, gae_lambda=0.95, max_new_tokens=50,):
        
        self.device = device
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.clip_epsilon = clip_epsilon

        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.max_new_tokens = max_new_tokens

        #1 . load tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = (self.tokenizer.eos_token)
        self.tokenizer.padding_side = "left"

        #2 Load policy
        self.policy = GPT2LMHeadModel.from_pretrained(model_name)

        checkpoint = torch.load(
            sft_checkpoint_path,
            map_location="cpu",
            weights_only=False
        )

        self.policy.load_state_dict(
            checkpoint['model_state_dict']
        )

        self.policy.to(self.device)
        self.policy.train()

        #3 Load reference model
        self.reference_model = GPT2LMHeadModel.from_pretrained(model_name)
        self.reference_model.load_state_dict(checkpoint['model_state_dict'])
        self.reference_model.to(self.device)
        self.reference_model.eval()

        for param in self.reference_model.parameters():
            param.requires_grad = False

        #4 load value model
        self.value_model = ValueModel(model_name=model_name,
                                      sft_checkpoint_path=sft_checkpoint_path)
        self.value_model.to(self.device)
        self.value_model.train()

        #5 reward model 
        self.reward_model = GPT2RewardModel(model_name, sft_checkpoint_path)

        reward_checkpoint = torch.load(
            reward_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        self.reward_model.load_state_dict(
            reward_checkpoint["model_state_dict"]
        )

        self.reward_model.to(self.device)
        self.reward_model.eval()

        for param in self.reward_model.parameters():
            param.requires_grad = False

        #6 optimizer for policy and value 
        self.policy_optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=1e-5,
        )

        self.value_optimizer = torch.optim.AdamW(
            self.value_model.parameters(),
            lr=1e-5,
        )

        #6. rollout
        self.rollout = Rollout(
            policy= self.policy,
            reference_model= self.reference_model,
            value_model= self.value_model,
            tokenizer= self.tokenizer,
            device = self.device
        )

        #7 Reward + KL
        self.reward_kl = RewardKL(reward_model=self.reward_model, kl_coef=kl_coef)

    # one ppo iteration
    def train_step(self, prompts):

        # ==================================================
        # 1. Rollout
        # ==================================================

        rollout_data = self.rollout.generate(
            prompts=prompts,
            max_new_tokens=self.max_new_tokens,
        )

        input_ids = rollout_data["input_ids"]
        response_ids = rollout_data["response_ids"]

        old_log_probs = rollout_data[
            "old_log_probs"
        ]

        ref_log_probs = rollout_data[
            "ref_log_probs"
        ]

        values = rollout_data["values"]

        attention_mask = rollout_data[
            "attention_mask"
        ]

        response_mask = rollout_data[
            "response_mask"
        ]

        # ==================================================
        # 2. Reward + KL
        # ==================================================

        reward_data = self.reward_kl.compute(
            input_ids=input_ids,
            old_log_probs=old_log_probs,
            ref_log_probs=ref_log_probs,
            attention_mask=attention_mask,
            response_mask=response_mask,
        )

        rewards = reward_data["rewards"]

        # ==================================================
        # 3. GAE
        # ==================================================

        advantages, returns = compute_gae(
            rewards=rewards,
            values=values,
            gamma=self.gamma,
            lam=self.gae_lambda,
        )

        # ==================================================
        # 4. Normalize advantages
        # ==================================================

        masked_advantages = (
            advantages * response_mask
        )

        token_count = (
            response_mask.sum()
            .clamp_min(1.0)
        )

        advantage_mean = (
            masked_advantages.sum()
            / token_count
        )

        advantage_variance = (
            (
                (
                    masked_advantages
                    - advantage_mean
                )
                * response_mask
            )
            .pow(2)
            .sum()
            / token_count
        )

        advantages = (
            advantages - advantage_mean
        ) / torch.sqrt(
            advantage_variance + 1e-8
        )

        # ==================================================
        # 5. Current policy forward pass
        # ==================================================

        outputs = self.policy(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        logits = outputs.logits

        # ==================================================
        # 6. New log probabilities
        # ==================================================

        log_probs = torch.log_softmax(
            logits,
            dim=-1,
        )

        shifted_logits = log_probs[
            :, :-1, :
        ]

        shifted_tokens = input_ids[
            :, 1:
        ]

        token_log_probs = torch.gather(
            shifted_logits,
            dim=-1,
            index=shifted_tokens.unsqueeze(-1),
        ).squeeze(-1)

        # Response starts after the padded prompt.
        prompt_padded_length = (
            input_ids.shape[1]
            - response_ids.shape[1]
        )

        new_log_probs = token_log_probs[
            :,
            prompt_padded_length - 1:
        ]

        # ==================================================
        # 7. PPO policy loss
        # ==================================================

        policy_loss = compute_policy_loss(
            new_log_probs=new_log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages.detach(),
            clip_epsilon=self.clip_epsilon,
            mask=response_mask,
        )

        # ==================================================
        # 8. Entropy
        # ==================================================

        response_logits = logits[
            :,
            prompt_padded_length - 1:-1,
            :
        ]

        entropy = compute_entropy(
            response_logits,
            mask=response_mask,
        )

        policy_loss_total = (
            policy_loss
            - self.entropy_coef * entropy
        )

        # ==================================================
        # 9. Policy update
        # ==================================================

        self.policy_optimizer.zero_grad()

        policy_loss_total.backward()

        self.policy_optimizer.step()

        # ==================================================
        # 10. Value model
        # ==================================================

        new_values = self.value_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        new_values = new_values[
            :,
            prompt_padded_length:
        ]

        value_loss = compute_value_loss(
            values=new_values,
            returns=returns.detach(),
            mask=response_mask,
        )

        value_loss_total = (
            self.value_coef * value_loss
        )

        # ==================================================
        # 11. Value update
        # ==================================================

        self.value_optimizer.zero_grad()

        value_loss_total.backward()

        self.value_optimizer.step()

        # ==================================================
        # 12. Metrics
        # ==================================================

        mean_kl = (
            (
                reward_data["kl"]
                * response_mask
            ).sum()
            / response_mask.sum().clamp_min(1.0)
        )

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "rm_reward": (
                reward_data["rm_reward"]
                .mean()
                .item()
            ),
            "mean_kl": mean_kl.item(),
        }