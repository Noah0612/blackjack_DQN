import torch
import torch.optim as optim
import numpy as np
import random
from agents.dqn_agent import DQNAgent, DQN
from config import DEVICE

class DoubleDQNAgent(DQNAgent):
    def __init__(self, env, config):
        super().__init__(env, config)
        self.target_model = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.target_update_freq = self.config.get("TARGET_UPDATE", 10)

    def train(self):
        print("Starting Double DQN Training...")
        episodes = self.config["EPISODES"]
        
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
            
            if e % self.target_update_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())
            
            avg_loss = np.mean(episode_loss_list) if episode_loss_list else 0
            self.log_metrics(e, episode_reward, avg_loss, self.epsilon)
            
            if (e+1) % 500 == 0:
                print(f"Episode {e+1}/{episodes} - Epsilon: {self.epsilon:.2f} - Reward: {episode_reward}")

        self.save("double_dqn_blackjack")

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

        # Normalize
        states = self.transform_state(states)
        next_states = self.transform_state(next_states)

        # Double DQN Logic
        with torch.no_grad():
            best_actions = self.model(next_states).argmax(1).unsqueeze(1)
            next_q_values = self.target_model(next_states).gather(1, best_actions)
            target_q = rewards + (1 - dones) * self.config["GAMMA"] * next_q_values
        
        current_q = self.model(states).gather(1, actions)
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.config["EPS_MIN"]:
            self.epsilon *= self.config["EPS_DECAY"]

        return loss.item()