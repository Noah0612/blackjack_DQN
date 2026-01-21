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
        
        # Stream 1: State Value V(s) -> Scalar
        self.value_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1) 
        )
        
        # Stream 2: Advantages A(s, a) -> Vector of size [Action_Dim]
        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim) 
        )

    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Aggregate: Q = V + (A - mean(A)) across dim=1
        q_vals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_vals

class DuelingDQNAgent(DQNAgent):
    def __init__(self, env, config):
        super().__init__(env, config)
        
        # Override Model and Target Model with Dueling Architecture
        self.model = DuelingNetwork(self.state_dim, self.action_dim).to(self.device)
        
        self.target_model = DuelingNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["LR"])