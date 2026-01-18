import torch
import torch.nn as nn
import torch.optim as optim
from agents.dqn_agent import DQNAgent
from config import DEVICE

class DuelingNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingNetwork, self).__init__()
        
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_vals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_vals

class DuelingDQNAgent(DQNAgent):
    def __init__(self, env, config):
        # Pass config to DQNAgent, which passes it to BaseAgent
        super().__init__(env, config)
        
        # Override the model
        self.model = DuelingNetwork(self.state_dim, self.action_dim).to(self.device)
        
        # Re-attach optimizer using the LR from the config
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["LR"])