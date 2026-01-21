import numpy as np
import collections
from agents.base_agent import BaseAgent

class ValueIterationAgent(BaseAgent):
    def __init__(self, env, config):
        super().__init__(env, config)
        self.gamma = self.config["GAMMA"]
        self.theta = self.config["THETA"]
        self.max_iterations = self.config["MAX_ITERATIONS"]
        
        # Initialize V(s) = 0 and Q(s,a) = 0
        self.V = collections.defaultdict(float)
        self.q_table_dict = collections.defaultdict(lambda: np.zeros(self.env.action_space.n))

    def train(self):
        print("Starting Value Iteration Training...")
        
        env_unwrapped = self.env.unwrapped
        if hasattr(env_unwrapped, "P"):
            P = env_unwrapped.P
        else:
            raise NotImplementedError("Environment P table not found.")
        
        states = list(P.keys())
        
        for i in range(self.max_iterations):
            delta = 0
            
            for s in states:
                v = self.V[s]
                
                # Compute Q(s,a) for all a using current V
                action_values = []
                for a in range(self.env.action_space.n):
                    transitions = P[s][a]
                    val = 0
                    for prob, next_s, reward, done in transitions:
                        next_val = 0 if done else self.V[next_s]
                        # Sum: prob * (r + gamma * V(s'))
                        val += prob * (reward + self.gamma * next_val)
                    action_values.append(val)
                
                # Update V(s) = max_a Q(s,a)
                best_action_value = max(action_values)
                self.V[s] = best_action_value
                
                delta = max(delta, abs(v - best_action_value))
            
            if (i+1) % 100 == 0:
                 print(f"Iteration {i+1}, Delta: {delta:.6f}")
                 
            if delta < self.theta:
                print(f"Value Iteration Converged at Iteration {i+1}!")
                break
        
        # Populate Final Q-Table from converged V
        print("Deriving Policy from V(s)...")
        for s in states:
            q_values = np.zeros(self.env.action_space.n)
            for a in range(self.env.action_space.n):
                transitions = P[s][a]
                val = 0
                for prob, next_s, reward, done in transitions:
                    next_val = 0 if done else self.V[next_s]
                    val += prob * (reward + self.gamma * next_val)
                q_values[a] = val
            self.q_table_dict[s] = q_values

        self.save_vi_table("vi_table")

    def get_action(self, state):
        # Convert tensor (from plotting) or ndarray to tuple key
        if hasattr(state, "cpu"): 
             # Logic to handle potential tensor inputs if plotting scripts sends them
            pass 
        
        if isinstance(state, list): 
             state = tuple(state)
        if isinstance(state, np.ndarray):
             state = tuple(state.tolist())

        if state not in self.q_table_dict:
            return self.env.action_space.sample()
            
        return int(np.argmax(self.q_table_dict[state]))

    def save_vi_table(self, filename):
        import pickle
        import os
        if not os.path.exists('models'):
            os.makedirs('models')
        with open(f"models/{filename}.pkl", "wb") as f:
            pickle.dump(self.q_table_dict, f)
        print(f"VI Table saved to models/{filename}.pkl")