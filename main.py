import argparse
import gymnasium as gym
from agents.dqn_agent import DQNAgent
from agents.dueling_dqn_agent import DuelingDQNAgent
from agents.double_dqn_agent import DoubleDQNAgent
from agents.value_iteration_agent import ValueIterationAgent
from agents.ppo_agent import PPOAgent
from utils.plotting import plot_strategy, plot_ppo_strategy
from config import DQN_CONFIG, DUELING_CONFIG, DOUBLE_DQN_CONFIG, VI_CONFIG, PPO_CONFIG

def main(agent_type):
    # Initialize Environment
    env = gym.make('Blackjack-v1', sab=True)
    
    # Instantiate specific agent based on CLI arg
    if agent_type == "dqn":
        agent = DQNAgent(env, config=DQN_CONFIG)
        
    elif agent_type == "dueling":
        agent = DuelingDQNAgent(env, config=DUELING_CONFIG)
        
    elif agent_type == "double_dqn":
        agent = DoubleDQNAgent(env, config=DOUBLE_DQN_CONFIG)
        
    elif agent_type == "vi":
        agent = ValueIterationAgent(env, config=VI_CONFIG) 
        
    elif agent_type == "ppo":
        agent = PPOAgent(env, config=PPO_CONFIG)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    agent.train()
    
    # Visualize
    if agent_type == "ppo":
        plot_ppo_strategy(
            agent,
            usable_ace=False,
            save_dir="assets",
            filename="ppo_hard_strategy.png"
        )
        plot_ppo_strategy(
            agent,
            usable_ace=True,
            save_dir="assets",
            filename="ppo_soft_strategy.png"
        )
    
    plot_strategy(agent, agent_type)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="dqn", help="dqn, dueling, double_dqn, ppo, or vi")
    args = parser.parse_args()
    
    main(args.agent)