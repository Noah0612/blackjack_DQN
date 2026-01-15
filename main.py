import argparse
import gymnasium as gym
from agents.dqn_agent import DQNAgent
from utils.plotting import plot_strategy

def main(agent_type):
    # Initialize Environment
    env = gym.make('Blackjack-v1', sab=True)
    
    # Agent Factory
    if agent_type == "dqn":
        agent = DQNAgent(env)
    elif agent_type == "dueling":
        # Placeholder for future DuelingDQN
        raise NotImplementedError("Dueling DQN not implemented yet.")
    elif agent_type == "vi":
        # Placeholder for future Value Iteration
        raise NotImplementedError("Value Iteration not implemented yet.")
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    # Train
    agent.train()
    
    # Visualize
    plot_strategy(agent, agent_type, usable_ace=False, title="Hard Totals")
    plot_strategy(agent, agent_type, usable_ace=True, title="Soft Totals")

if __name__ == "__main__":
    # Allows running like: python main.py --agent dqn
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="dqn", help="dqn, dueling, or vi")
    args = parser.parse_args()
    
    main(args.agent)