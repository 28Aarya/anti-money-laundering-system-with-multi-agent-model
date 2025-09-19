import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.base_agent import BaseAgent


class NetworkAnalysisAgent(BaseAgent):
    def __init__(self, alpha, gamma, epsilon, algorithm, use_features=None):
        """
        Parameters:
            alpha (float): Learning rate
            gamma (float): Discount factor
            epsilon (float): Exploration rate
            algorithm (str): RL algorithm (e.g., 'Q-learning', 'SARSA')
            use_features (list or None): Features to use for state extraction
        """
        super().__init__(alpha, gamma, epsilon, algorithm)
        self.use_features = use_features or [
            'from_degree', 'to_degree',
            'sender_clustering', 'receiver_clustering',
            'from_betweenness', 'to_betweenness'
        ]

    def extract_state(self, window_transaction_dicts):
        """Aggregate graph features across the sliding window."""
        aggregated = {feature: 0.0 for feature in self.use_features}
        for tx in window_transaction_dicts:
            for feature in self.use_features:
                aggregated[feature] += tx.get(feature, 0.0)
        num_tx = len(window_transaction_dicts)
        return [aggregated[feature] / num_tx for feature in self.use_features]  # mean pooling

    def choose_action(self, window_transaction_dicts, log=False):
        state = self.extract_state(window_transaction_dicts)
        state_key = tuple(state)
        action = super().choose_action(state)
        if log:
            print(f"[NAA] Chose action {action} for window state {state_key}, Q-values: {self.q_table.get(state_key)}")
        return action

    def update(self, window_transaction_dicts, action, reward, next_window_transaction_dicts, next_action=None):
        state = self.extract_state(window_transaction_dicts)
        next_state = self.extract_state(next_window_transaction_dicts) if next_window_transaction_dicts else None
        self.update_q_table(state, action, reward, next_state, next_action)

    def decay_epsilon(self, decay_rate=0.995, min_epsilon=0.01):
        self.epsilon = max(self.epsilon * decay_rate, min_epsilon)
