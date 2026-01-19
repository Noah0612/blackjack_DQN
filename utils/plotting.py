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
    hard_totals = list(range(21, 3, -1))
    
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
                # We check specifically for neural network agents to use the model directly