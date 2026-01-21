import torch

# Device selection: CUDA if available, else CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Master switch to enable/disable TensorBoard logging globally
ENABLE_TENSORBOARD = True 

# --- Agent Configurations ---

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

PPO_CONFIG = {
    "EPISODES": 10000,
    "GAMMA": 0.99,
    "LR_ACTOR": 0.0003,
    "LR_CRITIC": 0.001,
    "K_EPOCHS": 4,
    "EPS_CLIP": 0.2,
    "UPDATE_TIMESTEP": 2000
}