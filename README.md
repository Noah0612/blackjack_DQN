# Blackjack Reinforcement Learning Agents

A research project implementing and comparing multiple deep reinforcement learning algorithms applied to the game of Blackjack. The goal of this repository is to explore how different RL architectures — Value Iteration, Deep Q-Network (DQN) and its variants (Double DQN, Dueling DQN) as well as Proximal Policy Optimization (PPO) — perform on a stochastic decision-making task.

## 🚀 Project Overview

In this project: 
We implement, train, and evaluate several RL methods:
* Value Iteration
* DQN
* Double DQN
* Dueling DQN
* PPO

We compare:
* learning stability
* final reward performance
* learned policies

This work combines theoretical foundations with practical experiments in Python.

## 📁 Repository Structure
.

├── agents/              # Implementation of RL agents 

├── utils/               # Utility functions (plotting)

├── models/              # Saved models / checkpoints

├── runs/                # Training logs & results

├── assets/              # Visuals / plots / figures

├── main.py              # Entry point for training/evaluation

├── config.py            # Configuration & hyperparameters

├── README.md            # Project documentation

├── requirements.txt     # Python dependencies


## 🧠 Algorithms Included
### Value Iteration 
Baseline algorithms for solving MDP's

### DQN (Deep Q-Network)

Value-based algorithm using a neural network to approximate Q-values.

Stabilization via experience replay and a target network.

 ### Double DQN

Improves on DQN by decoupling action selection from evaluation.

Reduces value overestimation inherent in vanilla DQN.

### Dueling DQN

Separates estimation of state value and advantage function.

Particularly useful when some actions have minimal relative impact at certain states.

### PPO (Proximal Policy Optimization)

Policy-based method that directly optimizes a parametrized policy.

Uses a clipped objective to constrain updates and improve stability.

## 🛠 Installation

### 1. Clone the repository 
```
git clone https://github.com/Noah0612/blackjack_DQN.git
cd blackjack_DQN
```

### 2.Create a virtual environment (optional but recommended):
```
python3 -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

### 3.Install dependencies:
```
pip install -r requirements.txt
```
## ▶️ Training & Evaluation

### Train a specific agent 
* modify *config.py* to change hyper parameters
*  optional : run ```tensorboard --logdir runs ``` in another terminal to start tensorboard logs
* run ``` python -m main --agent <your_agent> ``` when your_agent =  dqn, dueling, double_dqn, ppo, or vi

### See the results
* The logs of tensorboard writers are in the runs directory, under the name Your_Agent_timestamp
* The resulting policy is stored in the assets directory
* The model (if relevant) is stored in the models directory under the name your_agent.pth

## 📊 Results
The repository includes performance logs and policy visualizations. Examples show:

* Reward vs training episodes

* Training loss curves

* Policy heatmaps comparing agent decisions

These help illustrate the strengths and limitations of each algorithm and how architectural choices affect learning and final performance.

## References
The algorithms are based on the following papers : 

[1] V. Mnih, K. Kavukcuoglu, D. Silver, et al.
Human-level control through deep reinforcement learning.
arXiv:1312.5602, 2013.

[2] H. van Hasselt, A. Guez, and D. Silver.
Deep reinforcement learning with Double Q-learning.
arXiv:1509.06461, 2015.

[3] Z. Wang, T. Schaul, M. Hessel, H. van Hasselt, M. Lanctot, and N. de Freitas.
Dueling Network Architectures for Deep Reinforcement Learning.
arXiv:1511.06581, 2015.

[4] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov.
Proximal Policy Optimization Algorithms.
arXiv:1707.06347, 2017.



