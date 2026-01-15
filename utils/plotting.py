import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import os

def plot_strategy(agent, agent_type, usable_ace=False, title="Hard Totals"):
    """
    Visualizes the agent's strategy on a heatmap.
    Works for both Neural Networks (DQN) and Table-based agents (VI).
    """
    policy_grid = np.zeros((10, 10)) # Rows: Player 12-21, Cols: Dealer 1-10
    
    for player_sum in range(12, 22):
        for dealer_card in range(1, 11):
            state = (player_sum, dealer_card, int(usable_ace))
            
            # ADAPTER: Check if agent is Deep Learning or Table-based
            if agent_type in ['dqn', 'dueling']:
                # Prepare state for Neural Network
                state_input = [state[0], state[1], state[2]]
                state_tensor = torch.FloatTensor(state_input).to(agent.device)
                with torch.no_grad():
                    action = torch.argmax(agent.model(state_tensor)).item()
            else:
                # Direct lookup for Table-based agents
                action = agent.get_action(state, eval_mode=True)
                
            policy_grid[player_sum-12][dealer_card-1] = action

    # Plotting
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(policy_grid, linewidth=0.5, annot=True, cmap="coolwarm", 
                    xticklabels=range(1, 11), yticklabels=range(12, 22), cbar=False)
    ax.set_title(f"{agent_type.upper()} Strategy ({title})")
    ax.set_xlabel("Dealer Showing Card")
    ax.set_ylabel("Player Sum")
    
    # Custom Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='blue', label='Stick (0)'),
                       Patch(facecolor='red', label='Hit (1)')]
    ax.legend(handles=legend_elements, loc='upper right')
    
    if not os.path.exists('assets'):
        os.makedirs('assets')
    
    filename = f"assets/{agent_type}_strategy_{'soft' if usable_ace else 'hard'}.png"
    plt.savefig(filename)
    print(f"Saved {filename}")
    plt.close()