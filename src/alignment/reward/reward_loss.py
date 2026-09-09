# Bradley–Terry Preference Loss

import torch

def preference_loss(
        chosen_rewards:torch.Tensor, 
        rejected_rewards:torch.Tensor)->torch.Tensor:
    
    return -torch.nn.functional.logsigmoid(chosen_rewards - rejected_rewards).mean()
