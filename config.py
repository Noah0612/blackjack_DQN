import torch

# General Training Settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MASTER SWITCH ---
ENABLE_TENSORBOARD = True  # Set to False to disable logging
# ---------------------

# 1. DQN Configuration
DQN_CONFIG = {
    "EPISODES": 5000,
    "BATCH_SIZE": 64,
    "GAMMA": 0.95,
    "EPS_START": 1.0,
    "EPS_MIN": 0.01,
    "EPS_DECAY": 0.995,
    "LR": 0.001,
    "MEMORY_SIZE": 100000,
    "TARGET_UPDATE": 10
}

# 2. Dueling DQN Configuration (Tuned for stability)
DUELING_CONFIG = {
    "EPISODES": 50000,       # More episodes
    "BATCH_SIZE": 64,
    "GAMMA": 0.99,
    "EPS_START": 1.0,
    "EPS_MIN": 0.01,
    "EPS_DECAY": 0.9999,     # Slower decay
    "LR": 0.0001,            # Lower learning rate
    "MEMORY_SIZE": 100000,
    "TARGET_UPDATE": 10
}

# 3. Double DQN Configuration
DOUBLE_DQN_CONFIG = {
    "EPISODES": 10000,
    "BATCH_SIZE": 64,
    "GAMMA": 0.95,
    "EPS_START": 1.0,
    "EPS_MIN": 0.01,
    "EPS_DECAY": 0.9995,
    "LR": 0.001,
    "MEMORY_SIZE": 100000,
    "TARGET_UPDATE": 20
}

# 4. Future Configs
VI_CONFIG = {
    "GAMMA": 0.99,
    "THETA": 1e-8,
    "MAX_ITERATIONS": 10000
}

# --- 5. PPO Configuration ---
# PPO is an On-Policy Gradient method 
PPO_CONFIG = {
    "GAMMA": 0.99,
    "STATE_DIM": 3,
    "HIDDEN_DIM": 64,
    "POLICY_LR": 3e-5,  # Learning rate for policy
    "VALUE_LR": 1e-3,    # Learning rate for value function
    "LAMBDA": 0.95,     #parameter of GAE(lambda) advantage estimator
    "ENTROPY_COEF": 0.01, # for entropy bonus
    "k_POLICY_UPDATES": 100,  # small k in PPO algorithm
    "K_UPDATE_EPOCHS": 4,  # number of epochs per update (reuse data)
    "EPS_CLIP": 0.2,     # PPO Clipping parameter
    "VALUE_CLIP" : 0.2, #clipping value function 
    "MINIBATCH_SIZE": 256,  # Minibatch size for updates
    "TARGET_KL": 10000000,    # Target KL divergence for early stopping (use big number to "disable")
    "EPISODES_PER_BATCH": 2_000  # number of steps to collect per update |D_k|
}