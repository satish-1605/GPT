from src.models.gpt import GPT2
from src.utils.config import GPTConfig


config = GPTConfig()
model = GPT2(config)

total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("=" * 60)
print(f"Total parameters     : {total_params:,}")
print(f"Trainable parameters : {trainable_params:,}")
print("=" * 60)

# token embedding (vocab_size × d_model) = 5000 × 256 = 1,280,000
#pe embedding (max_seq_len × d_model) = 128 × 256 = 32,768
# attn param = (d_model * w) = 4×(256**2 + 256) = 263,168
#MLP 256 → 1024 → 256  -> fc1 256×1024+1024 + fc2 -> 1024×256+256 = 263,168+262,400=525,568
# LayerNorm parameters -> 2×512=1,024

#One GPT2Block
# Attention     263,168
# MLP           525,568
# LayerNorm       1,024
# ──────────────────────
# Total          789,760

# gpt num_layers = 6 -> 789,760×6=4,738,560
# Final LayerNorm = 256+256=512

#LM Head 256 → 5000
# 256×5000=1,280,000 + 5000 = 1,285,000 but due to weight tying its weight count is notcounted 
# seprately only bias is considered

# -> 5000 

#Expected total 
# Token embedding       1,280,000
# Position embedding       32,768
# Transformer blocks    4,738,560
# Final LayerNorm             512
# LM head bias             5,000
# ───────────────────────────────
# Total                  6,056,840

assert total_params == 6_056_840
assert trainable_params == 6_056_840

print("✅ Parameter count test passed!")