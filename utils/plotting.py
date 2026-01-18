import matplotlib
# 1. Force non-interactive backend for WSL/Server environments
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import os
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

def plot_strategy(agent, agent_type):
    print(f"--- Starting Strategy Visualization for {agent_type} ---")
    
    # Define Ranges (Standard Casino Order: High to Low)
    soft_totals = list(range(21, 11, -1)) 
    hard_totals = list(range(21, 11, -1))
    dealer_cards = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "A"]
    
    total_rows = len(soft_totals) + len(hard_totals)
    policy_grid = np.zeros((total_rows, len(dealer_cards)))
    text_grid = []
    yticklabels = []

    row_idx = 0

    def fill_section(totals, is_soft):
        nonlocal row_idx
        prefix = "S" if is_soft else "H"
        
        for player_sum in totals:
            yticklabels.append(f"{prefix} {player_sum}")
            row_text = []
            
            for j, dealer_card_str in enumerate(dealer_cards):
                dealer_val = 1 if dealer_card_str == "A" else int(dealer_card_str)
                state = (player_sum, dealer_val, int(is_soft))
                
                # Get Action
                if agent_type in ['dqn', 'dueling', 'double_dqn']:
                    state_input = [state[0], state[1], state[2]]
                    
                    # 2. FIX: Add batch dimension .unsqueeze(0)
                    # The Dueling Network expects input shape [Batch, Features] to calculate mean(dim=1)
                    state_tensor = torch.FloatTensor(state_input).unsqueeze(0).to(agent.device)
                    
                    with torch.no_grad():
                        q_values = agent.model(state_tensor)
                        action = torch.argmax(q_values).item()
                else:
                    action = agent.get_action(state)
                
                policy_grid[row_idx][j] = action
                row_text.append("H" if action == 1 else "S")
            
            text_grid.append(row_text)
            row_idx += 1

    # Build Grid
    fill_section(soft_totals, is_soft=True)
    fill_section(hard_totals, is_soft=False)

    # Visualization
    print("Generating Heatmap...")
    cmap = ListedColormap(['#4A90E2', '#D0021B']) # Blue=Stand, Red=Hit
    
    plt.figure(figsize=(12, 14))
    
    ax = sns.heatmap(policy_grid, 
                     linewidths=1.0,
                     linecolor='white',
                     annot=np.array(text_grid),
                     fmt="",
                     cmap=cmap,
                     cbar=False,
                     xticklabels=dealer_cards,
                     yticklabels=yticklabels)

    ax.set_title(f"Blackjack Master Strategy ({agent_type.upper()})", fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel("Dealer Upcard", fontsize=14, fontweight='bold')
    ax.set_ylabel("Player Hand", fontsize=14, fontweight='bold')
    
    sep_line_y = len(soft_totals)
    ax.hlines([sep_line_y], *ax.get_xlim(), colors='black', linewidth=4)

    legend_elements = [
        Patch(facecolor='#D0021B', edgecolor='white', label='Hit'),
        Patch(facecolor='#4A90E2', edgecolor='white', label='Stand')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.2, 1.01), fontsize=12)

    plt.tight_layout()
    
    # Save
    if not os.path.exists('assets'):
        os.makedirs('assets')
    
    filename = f"assets/{agent_type}_master_strategy.png"
    plt.savefig(filename, dpi=300)
    print(f"SUCCESS: Saved master chart to {filename}") # Confirm save
    plt.close()