import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from agents.base_agent import BaseAgent
from config import DEVICE

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

class DQNAgent(BaseAgent):
    def __init__(self, env, config):
        # Pass config to the BaseAgent
        super().__init__(env, config)
        
        self.state_dim = 3
        self.action_dim = env.action_space.n
        self.device = DEVICE
        
        # Access config via self.config (inherited from BaseAgent)
        self.memory = deque(maxlen=self.config["MEMORY_SIZE"])
        self.epsilon = self.config["EPS_START"]
        
        self.model = DQN(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["LR"])
        self.criterion = nn.MSELoss()

    def get_action(self, state, eval_mode=False):
        if isinstance(state, tuple):
             state = [state[0], state[1], int(state[2])]

        if not eval_mode and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return torch.argmax(self.model(state_t)).item()

    def train(self):
        print("Starting DQN Training...")
        episodes = self.config["EPISODES"]
        
        for e in range(episodes):
            state, _ = self.env.reset()
            state = [state[0], state[1], int(state[2])]
            done = False
            
            while not done:
                action = self.get_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                next_state_proc = [next_state[0], next_state[1], int(next_state[2])]
                
                self.memory.append((state, action, reward, next_state_proc, done))
                self.replay()
                
                state = next_state_proc
            
            if (e+1) % 500 == 0:
                print(f"Episode {e+1}/{episodes} - Epsilon: {self.epsilon:.2f}")

        self.save("dqn_blackjack")

    def replay(self):
        batch_size = self.config["BATCH_SIZE"]
        
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        current_q = self.model(states).gather(1, actions)
        next_q = self.model(next_states).max(1)[0].unsqueeze(1)
        
        target_q = rewards + (1 - dones) * self.config["GAMMA"] * next_q
        
        loss = self.criterion(current_q, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.config["EPS_MIN"]:
            self.epsilon *= self.config["EPS_DECAY"]