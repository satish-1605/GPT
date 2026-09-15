import torch.nn as nn
import torch
from transformers import GPT2Model, GPT2Config

class ValueModel(nn.Module):
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

        self.transformer.load_state_dict(
                transformer_state_dict,
                strict=True,
            )

        self.value_head = nn.Linear(config.n_embd, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        hidden_states = outputs.last_hidden_state

        values = self.value_head(hidden_states)
        return values.squeeze(-1)
