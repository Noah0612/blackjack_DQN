import argparse
import gymnasium as gym
from agents.dqn_agent import DQNAgent
from agents.dueling_dqn_agent import DuelingDQNAgent
from agents.double_dqn_agent import DoubleDQNAgent
from agents.value_iteration_agent import ValueIterationAgent
from utils.plotting import plot_strategy
from config import DQN_CONFIG, DUELING_CONFIG, DOUBLE_DQN_CONFIG, VI_CONFIG

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
        raise NotImplementedError("PPO not implemented yet.")
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    agent.train()
    
    plot_strategy(agent, agent_type)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="dqn", help="dqn, dueling, double_dqn, ppo, or vi")
    args = parser.parse_args()
    
    main(args.agent)