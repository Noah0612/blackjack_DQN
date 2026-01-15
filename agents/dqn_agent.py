import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from agents.base_agent import BaseAgent
from config import *

# The Neural Network
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

# The Agent
class DQNAgent(BaseAgent):
    def __init__(self, env):
        super().__init__(env)
        
        self.state_dim = 3 # Player Sum, Dealer Card, Usable Ace
        self.action_dim = env.action_space.n
        self.device = DEVICE
        
        # Replay Buffer & Hyperparameters
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPS_START
        
        # Neural Network Setup
        self.model = DQN(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.criterion = nn.MSELoss()

    def get_action(self, state, eval_mode=False):
        # Convert tuple state (if necessary) to list for tensor conversion
        if isinstance(state, tuple):
             state = [state[0], state[1], int(state[2])]

        # Epsilon-Greedy Logic
        if not eval_mode and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return torch.argmax(self.model(state_t)).item()

    def train(self):
        print("Starting DQN Training...")
        for e in range(EPISODES):
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
            
            # Print progress
            if (e+1) % 500 == 0:
                print(f"Episode {e+1}/{EPISODES} - Epsilon: {self.epsilon:.2f}")

        # Save the result using Parent class method
        self.save("dqn_blackjack")

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Q-Learning with Neural Networks
        current_q = self.model(states).gather(1, actions)
        next_q = self.model(next_states).max(1)[0].unsqueeze(1)
        target_q = rewards + (1 - dones) * GAMMA * next_q
        
        loss = self.criterion(current_q, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > EPS_MIN:
            self.epsilon *= EPS_DECAY