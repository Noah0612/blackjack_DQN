import torch

# General Training Settings
EPISODES = 5000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters (DQN)
BATCH_SIZE = 64
GAMMA = 0.95            # Discount factor
EPS_START = 1.0         # Starting exploration probability
EPS_MIN = 0.01          # Minimum exploration probability
EPS_DECAY = 0.995       # Decay rate per episode
LR = 0.001              # Learning rate
MEMORY_SIZE = 100000    # Replay buffer size