import torch

# Shared Settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. DQN Configuration ---
DQN_CONFIG = {
    "EPISODES": 5000,
    "BATCH_SIZE": 64,
    "GAMMA": 0.95,
    "EPS_START": 1.0,
    "EPS_MIN": 0.01,
    "EPS_DECAY": 0.995,
    "LR": 0.001,
    "MEMORY_SIZE": 100000
}

# --- 2. Dueling DQN Configuration ---
DUELING_CONFIG = {
    "EPISODES": 6000,
    "BATCH_SIZE": 64,
    "GAMMA": 0.99,
    "EPS_START": 1.0,
    "EPS_MIN": 0.01,
    "EPS_DECAY": 0.995,
    "LR": 0.0005,
    "MEMORY_SIZE": 100000
}

# --- 3. Double DQN Configuration ---
DOUBLE_DQN_CONFIG = {
    "EPISODES": 5000,
    "BATCH_SIZE": 64,
    "GAMMA": 0.95,
    "EPS_START": 1.0,
    "EPS_MIN": 0.01,
    "EPS_DECAY": 0.995,
    "LR": 0.001,
    "MEMORY_SIZE": 100000
}

# --- 4. Value Iteration (VI) Configuration ---
# VI is not a gradient-based method, so it doesn't need LR or Batch Size.
VI_CONFIG = {
    "GAMMA": 0.99,
    "THETA": 1e-8,       # Convergence threshold
    "MAX_ITERATIONS": 10000
}

# --- 5. PPO Configuration ---
# PPO is an On-Policy Gradient method with different hyperparameters.
PPO_CONFIG = {
    "EPISODES": 10000,   # PPO often needs more episodes
    "GAMMA": 0.99,
    "LR_ACTOR": 0.0003,  # Learning rate for policy
    "LR_CRITIC": 0.001,  # Learning rate for value function
    "K_EPOCHS": 4,       # Update epochs per rollout
    "EPS_CLIP": 0.2,     # PPO Clipping parameter
    "UPDATE_TIMESTEP": 2000 # Update policy every n timesteps
}