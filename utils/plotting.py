import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import os
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

def plot_strategy(agent, agent_type):
    print(f"--- Starting Strategy Visualization for {agent_type} ---")
    
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
                    
                    # Create Batch Tensor
                    state_tensor = torch.FloatTensor(state_input).unsqueeze(0).to(agent.device)
                    
                    # Apply Normalization
                    state_tensor = agent.transform_state(state_tensor)
                    
                    with torch.no_grad():
                        q_values = agent.model(state_tensor)
                        action = torch.argmax(q_values).item()
                else:
                    action = agent.get_action(state)
                
                policy_grid[row_idx][j] = action
                row_text.append("H" if action == 1 else "S")
            
            text_grid.append(row_text)
            row_idx += 1

    fill_section(soft_totals, is_soft=True)
    fill_section(hard_totals, is_soft=False)

    print("Generating Heatmap...")
    cmap = ListedColormap(['#4A90E2', '#D0021B'])
    
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
    
    if not os.path.exists('assets'):
        os.makedirs('assets')
    
    filename = f"assets/{agent_type}_master_strategy.png"
    plt.savefig(filename, dpi=300)
    print(f"SUCCESS: Saved master chart to {filename}")
    plt.close()


def plot_ppo_strategy(
    agent,
    usable_ace: bool,
    save_dir: str,
    filename: str,
    device=None
):
    """
    Plot PPO policy as a probability heatmap.
    Color intensity = P(Hit | state)
    Each cell also shows the numeric probability.
    """

    if device is None:
        device = agent.device

    # Blackjack conventions (same as your existing plot)
    player_sums = list(range(12, 22))      # 12 → 21
    dealer_cards = [2, 3, 4, 5, 6, 7, 8, 9, 10, 1]  # Ace = 1

    heatmap = np.zeros((len(player_sums), len(dealer_cards)))

    # ---- compute probabilities ----
    for i, ps in enumerate(player_sums):
        for j, dc in enumerate(dealer_cards):
            state = np.array([ps, dc, int(usable_ace)], dtype=np.float32)
            state_t = torch.tensor(state).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = agent.policy_net(state_t)
                probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

            # Convention: action 0 = Stand, 1 = Hit
            heatmap[i, j] = probs[1]   # P(Hit)

    # ---- plotting ----
    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(
        heatmap,
        cmap="RdBu_r",
        vmin=0.0,
        vmax=1.0,
        aspect="auto"
    )

    # Axis labels
    ax.set_xticks(np.arange(len(dealer_cards)))
    ax.set_xticklabels(["2", "3", "4", "5", "6", "7", "8", "9", "10", "A"])
    ax.set_yticks(np.arange(len(player_sums)))
    ax.set_yticklabels(player_sums)

    ax.set_xlabel("Dealer Upcard")
    ax.set_ylabel("Player Sum")

    title = "PPO Policy — P(Hit)"
    title += " (Usable Ace)" if usable_ace else " (No Usable Ace)"
    ax.set_title(title)

    # ---- annotate probabilities ----
    for i in range(len(player_sums)):
        for j in range(len(dealer_cards)):
            p = heatmap[i, j]
            ax.text(
                j,
                i,
                f"{p:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=9
            )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("P(Hit)")

    # ---- save ----
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

    print(f"[PLOT] PPO strategy saved to {path}")
