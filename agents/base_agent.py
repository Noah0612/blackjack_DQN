from abc import ABC, abstractmethod
import os
import torch

class BaseAgent(ABC):
    def __init__(self, env):
        self.env = env
        
    @abstractmethod
    def get_action(self, state):
        """Every agent must be able to choose an action."""
        pass

    @abstractmethod
    def train(self):
        """Every agent must have a training loop or calculation."""
        pass

    def save(self, filename):
        """Shared logic to save models."""
        if not os.path.exists('models'):
            os.makedirs('models')
        
        # If the agent has a neural network (PyTorch model)
        if hasattr(self, 'model'):
            torch.save(self.model.state_dict(), f"models/{filename}.pth")
            print(f"Model saved to models/{filename}.pth")
        
        # If the agent is table-based (like Q-Learning/VI)
        elif hasattr(self, 'q_table'):
            import numpy as np
            np.save(f"models/{filename}.npy", self.q_table)
            print(f"Table saved to models/{filename}.npy")