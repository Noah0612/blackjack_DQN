from math import e
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
from collections import deque
from agents.base_agent import BaseAgent
from config import PPO_CONFIG, DEVICE

""" PPO algorithm implementation 
We annotate the code with the lines corresponding to the steps in the PPO algorithm.
Reference: https://spinningup.openai.com/en/latest/algorithms/ppo.html
"""

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.fc(x)


class PPOAgent(BaseAgent):
    def __init__(self, env, config=PPO_CONFIG):
        super().__init__(env)
        self.config = config
        self.device = DEVICE
        input_dim = env.observation_space.shape[0]
        hidden_dim = config["HIDDEN_DIM"]

        #define the actor : policy
        policy_output_dim = env.action_space.n
        self.policy_net =MLP(
            input_dim,
            hidden_dim,
            policy_output_dim
        ).to(self.device)

        #define the critic : value function
        value_output_dim = 1
        self.value_net = MLP(
            input_dim,
            hidden_dim,
            value_output_dim
        ).to(self.device)

        self.policy_optimizer = optim.Adam(
            self.policy_net.parameters(), lr=config["POLICY_LR"])
        self.value_optimizer = optim.Adam(
            self.value_net.parameters(), lr=config["VALUE_LR"])


    def train(self):
        print("Starting PPO Training...")

        for k in range(self.config["k_POLICY_UPDATES"]):
            # collect trajectories using current policy --> step 3
            all_states, all_actions, all_rewards, all_old_log_probs, all_values = self._batch()
            # compute returns and advantages --> steps 4 and 5
            all_returns = []
            all_advantages = []
            for rewards, values in zip(all_rewards, all_values):
                returns = self._compute_returns(rewards)
                advantages = self._compute_advantages(
                    returns, torch.cat(values))
                all_returns.append(returns)
                all_advantages.append(advantages)

            # create minibatches for SGD
            results_dataset = TensorDataset(
                torch.FloatTensor(np.concatenate(all_states)).to(self.device),
                torch.LongTensor(np.concatenate(all_actions)).to(self.device),
                torch.FloatTensor(np.concatenate(
                    all_old_log_probs)).to(self.device),
                torch.FloatTensor(np.concatenate(all_returns)).to(self.device),
                torch.FloatTensor(np.concatenate(
                    all_advantages)).to(self.device)
            )
            dataloader = DataLoader(
                results_dataset,
                batch_size=self.config["MINIBATCH_SIZE"],
                shuffle=True
            )

            # update policy and value networks for K epochs --> steps 6 and 7
            for _ in range(self.config["K_UPDATE_EPOCHS"]):
                for batch_states, batch_actions, batch_old_log_probs, batch_returns, batch_advantages in dataloader:
                    # Get current policy log probs
                    logits = self.policy_net(batch_states)
                    dist = torch.distributions.Categorical(logits=logits)
                    batch_log_probs = dist.log_prob(batch_actions)
                    batch_advantages = batch_advantages.detach()  # no grad on advantages
                    # Compute surrogate loss
                    policy_loss = self._surrogate_loss(
                        batch_old_log_probs, batch_log_probs, batch_advantages)
                    # Update policy network
                    self.policy_optimizer.zero_grad()
                    policy_loss.backward()
                    self.policy_optimizer.step()
                    # Compute value loss
                    values = self.value_net(batch_states).squeeze()
                    value_loss = nn.MSELoss()(values, batch_returns)
                    # Update value network
                    self.value_optimizer.zero_grad()
                    value_loss.backward()
                    self.value_optimizer.step()
            
    """
    run the current policy on the environment to collect trajectories
    --> step 3 in the PPO algorithm
    """
    def _collect_trajectory(self):
        states = []
        actions = []
        rewards = []
        log_probs = []
        values = []
        state, _ = self.env.reset()
        done = False
        while not done:
            norm_state = super().transform_state(state)
            state_t = torch.FloatTensor(norm_state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                # Get action probabilities from *current* policy
                logits = self.policy_net(state_t)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                # Get value estimate from *current* value function
                value = self.value_net(state_t)

            # Step the environment
            next_state, reward, terminated, truncated, _ = self.env.step(action.item())
            done = terminated or truncated
            # Store the transition
            states.append(state)
            actions.append(action.item())
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            state = next_state
        return states, actions, rewards, log_probs, values

    """
    collect a batch of trajectories
    --> step 3 in the PPO algorithm, called D_k
    """
    def _batch(self):
        all_states = []
        all_actions = []
        all_rewards = []
        all_log_probs = []
        all_values = []
        # collect EPISODES_PER_BATCH trajectories with the same policy
        for _ in range(self.config["EPISODES_PER_BATCH"]):
            states, actions, rewards, log_probs, values = self._collect_trajectory()
            all_states.append(states)
            all_actions.append(actions)
            all_rewards.append(rewards)
            all_log_probs.append(log_probs)
            all_values.append(values)
        return all_states, all_actions, all_rewards, all_log_probs, all_values

    """
    calculate the rewards-to-go for each time step, 
    --> step 4 in the PPO algorithm
    """
    def _compute_returns(self, rewards):
        returns = []
        G = 0.0
        gamma = self.config["GAMMA"]
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        return torch.tensor(returns, dtype=torch.float32, device=self.device)

    """
    calculate the advantages using the value function estimates,
    --> step 5 in the PPO algorithm
    We use Monte Carlo estimates for the advantages since the episodes are finite and short.
    """
    def _compute_advantages(self, returns, values):
        advantages = returns - values.squeeze()
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages

    """
    calculate the surrogate loss with clipping,
    --> step 6 in the PPO algorithm
    """
    def _surrogate_loss(self, old_log_probs, log_probs, advantages):
        ratios = torch.exp(log_probs - old_log_probs.detach())
        surr1 = ratios * advantages
        surr2 = torch.clamp(
            ratios, 1 - self.config["EPS_CLIP"], 1 + self.config["EPS_CLIP"]) * advantages
        loss = -torch.min(surr1, surr2).mean()
        return loss
