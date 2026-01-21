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
        super().__init__(env, config)
        
        self.state_dim = 3
        self.action_dim = env.action_space.n
        self.device = DEVICE
        
        self.memory = deque(maxlen=self.config["MEMORY_SIZE"])
        self.epsilon = self.config["EPS_START"]
        
        # Networks (Policy & Target)
        self.model = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_model = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["LR"])
        self.criterion = nn.MSELoss()

    def get_action(self, state, eval_mode=False):
        if isinstance(state, tuple):
             state = [state[0], state[1], int(state[2])]

        if not eval_mode and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        
        # Normalize Input
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        state_t = self.transform_state(state_t) 
        
        with torch.no_grad():
            return torch.argmax(self.model(state_t)).item()

    def train(self):
        print(f"Starting {self.__class__.__name__} Training...")
        episodes = self.config["EPISODES"]
        target_update = self.config.get("TARGET_UPDATE", 10)
        
        for e in range(episodes):
            state, _ = self.env.reset()
            state = [state[0], state[1], int(state[2])]
            done = False
            
            episode_reward = 0
            episode_loss_list = []
            
            while not done:
                action = self.get_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                next_state_proc = [next_state[0], next_state[1], int(next_state[2])]
                
                self.memory.append((state, action, reward, next_state_proc, done))
                
                loss = self.replay()
                if loss is not None:
                    episode_loss_list.append(loss)
                
                state = next_state_proc
                episode_reward += reward
            
            if e % target_update == 0:
                self.target_model.load_state_dict(self.model.state_dict())
            
            # Log Metrics
            avg_loss = np.mean(episode_loss_list) if episode_loss_list else 0
            self.log_metrics(e, episode_reward, avg_loss, self.epsilon)

            if (e+1) % 500 == 0:
                print(f"Episode {e+1}/{episodes} - Epsilon: {self.epsilon:.2f} - Reward: {episode_reward} - Avg Loss: {avg_loss:.4f}")

        self.save("dqn_blackjack")

    def replay(self):
        batch_size = self.config["BATCH_SIZE"]
        if len(self.memory) < batch_size:
            return None
        
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Normalize Batch
        states = self.transform_state(states)
        next_states = self.transform_state(next_states)

        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * self.config["GAMMA"] * next_q
        
        current_q = self.model(states).gather(1, actions)
        
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.config["EPS_MIN"]:
            self.epsilon *= self.config["EPS_DECAY"]

        return loss.item()