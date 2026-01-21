from abc import ABC, abstractmethod
import os
import torch
from torch.utils.tensorboard import SummaryWriter
import datetime
from config import ENABLE_TENSORBOARD

class BaseAgent(ABC):
    def __init__(self, env, config):
        self.env = env
        self.config = config
        
        # --- Global TensorBoard Logic ---
        if ENABLE_TENSORBOARD:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            agent_name = self.__class__.__name__
            log_dir = os.path.join("runs", f"{agent_name}_{current_time}")
            self.writer = SummaryWriter(log_dir=log_dir)
            print(f"TensorBoard logging enabled in: {log_dir}")
        else:
            self.writer = None
            print("TensorBoard logging is DISABLED (Global Switch).")
    
    def transform_state(self, state_tensor):
        """
        Normalizes the state tensor.
        Idx 0 (Player Sum): Divided by 21.0
        Idx 1 (Dealer Card): Divided by 10.0
        """
        norm_state = state_tensor.clone()
        norm_state[:, 0] = norm_state[:, 0] / 21.0
        norm_state[:, 1] = norm_state[:, 1] / 10.0
        return norm_state

    def log_metrics(self, episode, reward, loss, epsilon):
        if self.writer:
            self.writer.add_scalar("Reward/Episode", reward, episode)
            self.writer.add_scalar("Loss/Episode", loss, episode)
            self.writer.add_scalar("Epsilon", epsilon, episode)

    @abstractmethod
    def get_action(self, state):
        pass

    @abstractmethod
    def train(self):
        pass

    def save(self, filename):
        if not os.path.exists('models'):
            os.makedirs('models')
        
        if hasattr(self, 'model'):
            torch.save(self.model.state_dict(), f"models/{filename}.pth")
            print(f"Model saved to models/{filename}.pth")
        
        elif hasattr(self, 'q_table'):
            import numpy as np
            np.save(f"models/{filename}.npy", self.q_table)
            print(f"Table saved to models/{filename}.npy")