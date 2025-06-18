import random, torch, numpy as np

# Use same seed everywhere to ensure reproducibility
SEED = 8343

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
