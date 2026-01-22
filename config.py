import torch

# Device selection: CUDA if available, else CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Master switch to enable/disable TensorBoard logging globally
ENABLE_TENSORBOARD = True 

# --- Agent Configurations ---

DQN_CONFIG = {
    "EPISODES": 50000,
    "BATCH_SIZE": 256,
    "GAMMA": 1,
    "EPS_START": 1.0,
    "EPS_MIN": 0.01,
    "EPS_DECAY": 0.995,
    "LR": 0.003,
    "MEMORY_SIZE": 100000,
    "TARGET_UPDATE": 500
}

DUELING_CONFIG = {
    "EPISODES": 50000,
    "BATCH_SIZE": 64,
    "GAMMA": 0.99,
    "EPS_START": 1.0,
    "EPS_MIN": 0.01,
    "EPS_DECAY": 0.9999,
    "LR": 0.0001,
    "MEMORY_SIZE": 100000,
    "TARGET_UPDATE": 10
}

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
    "POLICY_LR": 3e-4,  # Learning rate for policy
    "VALUE_LR": 6e-4,    # Learning rate for value function
    "LAMBDA": 0.95,     #parameter of GAE(lambda) advantage estimator
    "ENTROPY_COEF": 0.015, # for entropy bonus
    "k_POLICY_UPDATES": 1_000,  # small k in PPO algorithm
    "K_UPDATE_EPOCHS": 4,  # number of epochs per update (reuse data)
    "EPS_CLIP": 0.2,     # PPO Clipping parameter
    "VALUE_CLIP" : 0.2, #clipping value function 
    "MINIBATCH_SIZE": 512,  # Minibatch size for updates
    "TARGET_KL": 100000,    # Target KL divergence for early stopping (use big number to "disable")
    "EPISODES_PER_BATCH": 4_096  # number of steps to collect per update |D_k|
}