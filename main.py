import argparse
import gymnasium as gym
from agents.dqn_agent import DQNAgent
from agents.dueling_dqn_agent import DuelingDQNAgent
from agents.double_dqn_agent import DoubleDQNAgent
from utils.plotting import plot_strategy

# Import all configs
from config import (
    DQN_CONFIG, 
    DUELING_CONFIG, 
    DOUBLE_DQN_CONFIG, 
    VI_CONFIG, 
    PPO_CONFIG
)

def main(agent_type):
    # Initialize Environment
    env = gym.make('Blackjack-v1', sab=True)
    
    # Agent Factory
    if agent_type == "dqn":
        agent = DQNAgent(env, config=DQN_CONFIG)
        
    elif agent_type == "dueling":
        agent = DuelingDQNAgent(env, config=DUELING_CONFIG)
        
    elif agent_type == "double":
        agent = DoubleDQNAgent(env, config=DOUBLE_DQN_CONFIG)
        
    elif agent_type == "ppo":
        # Placeholder for future PPOAgent
        # agent = PPOAgent(env, config=PPO_CONFIG)
        raise NotImplementedError("PPO not implemented yet.")
        
    elif agent_type == "vi":
        # Placeholder for future ValueIterationAgent
        # agent = ValueIterationAgent(env, config=VI_CONFIG)
        raise NotImplementedError("Value Iteration not implemented yet.")
        
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    # Train
    agent.train()
    
    # Visualize
    plot_strategy(agent, agent_type)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Updated help string to reflect future capabilities
    parser.add_argument("--agent", type=str, default="dqn", 
                        help="dqn, dueling, double, ppo, or vi")
    args = parser.parse_args()
    
    main(args.agent)