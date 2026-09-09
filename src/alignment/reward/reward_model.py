from transformers import GPT2Model, GPT2Config
import torch.nn as nn
import torch

class GPT2RewardModel(nn.Module):
    def __init__(self, model_name, sft_checkpoint_path):
        super().__init__()

        config = GPT2Config.from_pretrained(model_name)
        self.transformer = GPT2Model(config)

        checkpoint = torch.load(
            sft_checkpoint_path,
            map_location="cpu",
            weights_only= False
        )

        sft_state_dict = checkpoint["model_state_dict"]

        # Keep only Transformer weights
        transformer_state_dict = {
            key.replace("transformer.", "", 1) : value
            for key, value in sft_state_dict.items()
            if key.startswith("transformer.")
        }

        # Load SFT weights into Transformer backbone
        missing_keys, unexpected_keys = self.transformer.load_state_dict(
            transformer_state_dict,
            strict=True
        )

        if missing_keys:
            raise RuntimeError(
                f"Missing transformer keys: {missing_keys}"
            )

        if unexpected_keys:
            raise RuntimeError(
                f"Unexpected transformer keys: {unexpected_keys}"
            )
        
        self.reward_head = nn.Linear(config.n_embd, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        hidden_states = outputs.last_hidden_state
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(hidden_states.size(0), device = hidden_states.device)

        last_hidden_state = hidden_states[batch_indices, sequence_lengths]

        reward = self.reward_head(last_hidden_state)
        return reward.squeeze(-1)

        




