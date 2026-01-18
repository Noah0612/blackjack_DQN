import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import os
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

def plot_strategy(agent, agent_type):
    """
    Generates a single, combined professional strategy chart.
    Stacks Soft Totals (Top) and Hard Totals (Bottom).
    """
    # --- 1. Define Ranges (Standard Casino Order: High to Low) ---
    # Soft: 21 down to 12 (A+A)
    soft_totals = list(range(21, 11, -1)) 
    # Hard: 21 down to 4
    hard_totals = list(range(21, 11, -1))
    
    dealer_cards = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "A"]
    
    # We will build a single matrix for the heatmap
    # Dimensions: (Rows_Soft + Rows_Hard) x (10 Dealer Cards)
    total_rows = len(soft_totals) + len(hard_totals)
    policy_grid = np.zeros((total_rows, len(dealer_cards)))
    text_grid = [] # Stores "H" or "S"
    yticklabels = [] # Stores "S 20", "H 18", etc.

    row_idx = 0

    # --- 2. Helper Function to Fill Rows ---
    def fill_section(totals, is_soft):
        nonlocal row_idx
        prefix = "S" if is_soft else "H"
        
        for player_sum in totals:
            yticklabels.append(f"{prefix} {player_sum}")
            row_text = []
            
            for j, dealer_card_str in enumerate(dealer_cards):
                # Convert "A" to 1 for the environment
                dealer_val = 1 if dealer_card_str == "A" else int(dealer_card_str)
                state = (player_sum, dealer_val, int(is_soft))
                
                # Get Action
                if agent_type in ['dqn', 'dueling']:
                    state_input = [state[0], state[1], state[2]]
                    state_tensor = torch.FloatTensor(state_input).to(agent.device)
                    with torch.no_grad():
                        q_values = agent.model(state_tensor)
                        action = torch.argmax(q_values).item()
                else:
                    action = agent.get_action(state)
                
                policy_grid[row_idx][j] = action
                row_text.append("H" if action == 1 else "S")
            
            text_grid.append(row_text)
            row_idx += 1

    # --- 3. Build the Master Grid ---
    # Section 1: Soft Totals (Top)
    fill_section(soft_totals, is_soft=True)
    
    # Section 2: Hard Totals (Bottom)
    fill_section(hard_totals, is_soft=False)

    # --- 4. Visualization ---
    # Colors: Blue=Stand (0), Red=Hit (1)
    cmap = ListedColormap(['#4A90E2', '#D0021B'])
    
    plt.figure(figsize=(12, 14)) # Taller figure to fit everything
    
    ax = sns.heatmap(policy_grid, 
                     linewidths=1.0,
                     linecolor='white',
                     annot=np.array(text_grid),
                     fmt="",
                     cmap=cmap,
                     cbar=False,
                     xticklabels=dealer_cards,
                     yticklabels=yticklabels)

    # Styling
    ax.set_title(f"Blackjack Master Strategy ({agent_type.upper()})", fontsize=20, fontweight='bold', pad=20)
    ax.set_xlabel("Dealer Upcard", fontsize=14, fontweight='bold')
    ax.set_ylabel("Player Hand", fontsize=14, fontweight='bold')
    
    # Add a horizontal line to separate Soft and Hard sections visually
    # The line should be after the last Soft row
    sep_line_y = len(soft_totals)
    ax.hlines([sep_line_y], *ax.get_xlim(), colors='black', linewidth=4)

    # Custom Legend
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
    print(f"Saved master chart to {filename}")
    plt.close()