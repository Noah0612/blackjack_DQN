import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from agents.base_agent import BaseAgent
from config import PPO_CONFIG, DEVICE
import os

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
        super().__init__(env, config)
        self.config = config
        self.device = DEVICE
        input_dim = config["STATE_DIM"]
        hidden_dim = config["HIDDEN_DIM"]

        self.total_env_steps = 0
        self.total_ep = 0

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


        #load preexisting policy. comment out if you want full training from scratch
        #DEBUG
        if os.path.exists("models/ppo.pth"):
            self.load("models/ppo.pth")
            print("Re using previous policy")
            return

        print("Starting PPO Training...")

        for k in range(self.config["k_POLICY_UPDATES"]):

            print(
                f"--- Policy Update {k+1}/{self.config['k_POLICY_UPDATES']} ---")

            # collect trajectories using current policy --> step 3
            all_states, all_actions, all_rewards, all_old_log_probs, all_values = self._batch()

            #compute and log average rewards for evaluation
            all_rewards_cat = np.concatenate(all_rewards)
            if self.writer:
                self.writer.add_scalar("Eval/Rewards", all_rewards_cat.mean(), self.total_ep)

            # compute returns and advantages --> steps 4 and 5
            all_returns = []
            all_advantages = []
            for rewards, values in zip(all_rewards, all_values):
                # returns = self._compute_returns(rewards)
                # advantages = self._compute_advantages(
                #     returns, torch.cat(values))
                advantages, returns = self._compute_gae(rewards, values)
                advantages = torch.clamp(advantages, -5.0, 5.0)
                all_returns.append(returns)
                all_advantages.append(advantages)

            # normalize advantages
            all_advantages = torch.cat(all_advantages).to(self.device)
            if all_advantages.numel() > 1:
                all_advantages = (all_advantages - all_advantages.mean()) / \
                     (all_advantages.std(unbiased=False) + 1e-8)
                if self.writer: 
                    self.writer.add_scalar("Debug/AdvStd", advantages.std().item(), self.total_env_steps) 
            else:
                all_advantages = all_advantages - all_advantages.mean()


            # create minibatches for SGD
            results_dataset = TensorDataset(
                torch.FloatTensor(np.concatenate(all_states)).to(self.device),
                torch.LongTensor(np.concatenate(all_actions)).to(self.device),
                torch.FloatTensor(np.concatenate(
                    all_old_log_probs)).to(self.device),
                torch.FloatTensor(np.concatenate(all_returns)).to(self.device),
                torch.FloatTensor(all_advantages).to(self.device),
                torch.FloatTensor(np.concatenate([v[:-1] for v in all_values])).to(self.device)
            )
            dataloader = DataLoader(
                results_dataset,
                batch_size=self.config["MINIBATCH_SIZE"],
                shuffle=True
            )

            # update policy and value networks for K epochs --> steps 6 and 7
            for epoch in range(self.config["K_UPDATE_EPOCHS"]):
                stop = False
                for batch_states, batch_actions, batch_old_log_probs, batch_returns, batch_advantages, batch_values in dataloader:
                    # Get current policy log probs
                    logits = self.policy_net(batch_states)
                    dist = torch.distributions.Categorical(logits=logits)
                    entropy = dist.entropy().mean()
                    batch_log_probs = dist.log_prob(batch_actions)
                    batch_advantages = batch_advantages.detach()  # no grad on advantages
                    # Compute surrogate loss
                    policy_loss, surrogate, approx_kl, clip_frac = self._compute_loss(
                        batch_old_log_probs, 
                        batch_log_probs, 
                        batch_advantages,
                        entropy
                        )

                    #update policy 
                    self.policy_optimizer.zero_grad()
                    policy_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
                    self.policy_optimizer.step()

                    with torch.no_grad():
                        new_logits = self.policy_net(batch_states)
                        new_dist = torch.distributions.Categorical(logits=new_logits)
                        new_log_probs = new_dist.log_prob(batch_actions)
                        approx_kl = (batch_old_log_probs - new_log_probs).mean()

                    if approx_kl > self.config["TARGET_KL"]:
                        print(f"Early stopping at epoch {epoch} due to KL.")
                        stop = True
                        break

                    # Compute value loss
                    value_pred = self.value_net(batch_states).squeeze()
                    value_pred_old = batch_values.detach()

                    value_clipped = value_pred_old + torch.clamp(
                        value_pred - value_pred_old,
                        -self.config["VALUE_CLIP"],
                        self.config["VALUE_CLIP"]
                    )

                    corr = torch.corrcoef(
                        torch.stack([batch_advantages, batch_returns - value_pred_old])
                    )[0,1]

                    value_loss = torch.max(
                        (value_pred - batch_returns) ** 2,
                        (value_clipped - batch_returns) ** 2
                    ).mean()


                    # Update value network
                    self.value_optimizer.zero_grad()
                    value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
                    self.value_optimizer.step()


            
                    #logging 
                    if self.writer:
                        self.writer.add_scalar(
                            "Loss/Policy", policy_loss.item(), self.total_env_steps)
                        self.writer.add_scalar(
                            "Loss/Value", value_loss.item(), self.total_env_steps)
                        self.writer.add_scalar(
                            "Surrogate", surrogate.item(), self.total_env_steps)
                        self.writer.add_scalar(
                            "Approx_KL", approx_kl.item(), self.total_env_steps)
                        self.writer.add_scalar(
                            "Clip_Fraction", clip_frac.item(), self.total_env_steps)
                        self.writer.add_scalar(
                            "Entropy", entropy.item(), self.total_env_steps)
                        self.writer.add_scalar(
                            "Debug/Corr", corr.item(), self.total_env_steps)

                if stop :
                    break

        self.save("ppo")

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
            raw_state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            state_t = super().transform_state(raw_state_t)
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
            states.append(state_t)
            actions.append(action.item())
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value.detach().squeeze())
            state = next_state

        # bootstrap value for terminal state
        with torch.no_grad():
            raw_state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            state_t = super().transform_state(raw_state_t)
            last_value = self.value_net(state_t)

        values.append(last_value.detach().squeeze())

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
            self.total_ep += 1
        self.total_env_steps += sum(len(r) for r in all_rewards)
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
        return advantages


    """
    alternate version : use GAE(lambda) estimator https://arxiv.org/abs/1506.02438
    """
    def _compute_gae(self, rewards, values):
        gamma = self.config["GAMMA"]
        lam = self.config["LAMBDA"]


        rewards = torch.tensor(rewards, dtype=torch.float32, device=values[0].device)
        values = torch.stack(values)  # shape [T+1]

        T = rewards.shape[0]
        advantages = torch.zeros(T, device=values.device)

        gae = 0.0
        for t in reversed(range(T)):
            delta = rewards[t] + gamma * values[t + 1] - values[t]
            gae = delta + gamma * lam * gae
            advantages[t] = gae

        returns = advantages + values[:-1]  
        return advantages, returns


        


    """
    calculate the surrogate loss with clipping,
    --> step 6 in the PPO algorithm
    optionnaly add an entropy bonus
    """
    def _compute_loss(self, old_log_probs, log_probs, advantages, entropy):
        ratios = torch.exp(log_probs - old_log_probs.detach())
        if self.writer :
            self.writer.add_scalar("Debug/RatioMean", ratios.mean().item(), self.total_env_steps)
            self.writer.add_scalar("Debug/RatioStd", ratios.std().item(), self.total_env_steps)

        surr1 = ratios * advantages
        surr2 = torch.clamp(
            ratios, 1 - self.config["EPS_CLIP"], 1 + self.config["EPS_CLIP"]) * advantages
        surrogate = torch.min(surr1, surr2)
        entropy_bonus = self.config["ENTROPY_COEF"] * entropy
        loss = -(surrogate + entropy_bonus).mean()
        approx_kl = (old_log_probs - log_probs).mean()
        clip_frac = ((ratios - 1.0).abs() >
                     self.config["EPS_CLIP"]).float().mean()
        return loss, surrogate.mean(), approx_kl, clip_frac

    """
    get action from policy for a given state
    --> used for evaluation
    """
    def get_action(self, state):
        raw_state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        state_t = super().transform_state(raw_state_t)
        with torch.no_grad():
            logits = self.policy_net(state_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
        return action.item()

    def save(self, filename):
        if not os.path.exists('models'):
            os.makedirs('models')
        
        # Saves state_dict for PyTorch models or npy array for tables
        torch.save(self.policy_net.state_dict(), f"models/{filename}.pth")
        print(f"Policy saved to models/{filename}.pth")

    def load(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint)
        print(f"Policy load from checkpoint {path}")
        