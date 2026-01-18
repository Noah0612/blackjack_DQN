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
        raise NotImplementedError("Dueling DQN not implemented yet.")
    elif agent_type == "vi":
        raise NotImplementedError("Value Iteration not implemented yet.")
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    # Train
    agent.train()
    
    # Visualize
    # FIX: We now call this ONCE, with no extra arguments.
    # It generates the single "Master Strategy" table automatically.
    plot_strategy(agent, agent_type)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="dqn", help="dqn, dueling, or vi")
    args = parser.parse_args()
    
    main(args.agent)