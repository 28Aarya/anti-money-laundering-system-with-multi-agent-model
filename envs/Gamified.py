import random
import time
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.standardenv import LaunderingEnv


class GamifiedEnv(LaunderingEnv):  # Inheriting from StandardEnv
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Initialize the parent class
        
        # Add any additional initialization for the gamified features
        self.time_step = 0  # Track time steps
        self.difficulty_factor = 1.0  # Difficulty factor that increases over time
        self.fraud_patterns = ['smurfing', 'structuring', 'layering', 'rounding']
        
        # Introduce a way to track evolving fraud patterns
        self.active_fraud_patterns = random.sample(self.fraud_patterns, k=2)
        # Compute threshold for large AmountReceived (95th percentile)
        self.amount_bonus_threshold = np.percentile(self.data['AmountReceived'], 92)
        
    def increase_difficulty(self):
        # Increase difficulty over time by reducing reward for late detections, adding new fraud patterns
        self.time_step += 1
        if self.time_step % 10 == 0:  # Every 10 time steps, increase difficulty
            self.difficulty_factor += 0.1  # Increasing the complexity of detecting fraud
        
            # Randomly evolve the fraud patterns in play
            self.active_fraud_patterns = random.sample(self.fraud_patterns, k=3)
    
    def get_current_fraud_pattern(self):
        # Return a pattern that's in play depending on the difficulty factor
        return random.choice(self.active_fraud_patterns)
    
    def _compute_reward(self, txn, label, decision1, decision2, tma_confidence, naa_confidence):
        reward = super()._compute_reward(txn, label, decision1, decision2, tma_confidence, naa_confidence)
        
        # Evolving difficulty: Reward decreases over time, and new fraud patterns evolve
        if self.time_step > 10:  # After 10 time steps, increase the penalty for missed fraud detections
            reward -= 0.5 * self.difficulty_factor
        
        # Dynamic reward bonuses for faster flagging
        if self.time_step < 5:
            if txn['AmountReceived'] > self.amount_bonus_threshold:
                reward += 3.0  # Give a small bonus if they catch large transactions early
        return reward
    
    def step(self, tma_agent=None, naa_agent=None, tma_input=None, naa_input=None):
        # Standard step from the base environment (multi-agent compatible)
        state, reward, done, info = super().step(
            tma_agent=tma_agent,
            naa_agent=naa_agent,
            tma_input=tma_input,
            naa_input=naa_input
        )
        # Introduce evolving fraud pattern and adjust difficulty
        self.increase_difficulty()
        return state, reward, done, info

    def reset(self):
        # Reset any extra gamified state here
        self.time_step = 0
        self.difficulty_factor = 1.0
        self.active_fraud_patterns = random.sample(self.fraud_patterns, k=2)
        return super().reset()  # Reset the base environment
