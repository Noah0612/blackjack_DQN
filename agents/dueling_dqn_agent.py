import torch
import torch.nn as nn
import torch.optim as optim
from agents.dqn_agent import DQNAgent # Inherit logic from DQN
from config import *

class DuelingNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingNetwork, self).__init__()
        
        # Shared Feature Layer
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Stream 1: Value (V) - estimates how good the state is
        self.value_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Outputs 1 single value
        )
        
        # Stream 2: Advantage (A) - estimates benefit of each action
        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim) # Outputs value for every action
        )

    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine them: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_vals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_vals

class DuelingDQNAgent(DQNAgent):
    def __init__(self, env):
        # 1. Initialize the parent DQN Agent
        super().__init__(env)
        
        # 2. OVERRIDE the model with our new Dueling Network
        self.model = DuelingNetwork(self.state_dim, self.action_dim).to(self.device)
        
        # 3. Re-attach optimizer to the new parameters
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        
    # The rest of the training logic remains the same as DQNAgent