import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.base_agent import BaseAgent

class TransactionMonitoringAgent(BaseAgent):
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
            'AmountReceived', 'AmountPaid',
            'ReceivingCurrency_', 'PaymentCurrency_', 'PaymentFormat_'
        ]

    def extract_state(self, transaction_dict):
        state = []
        for f in self.use_features:
            if f.endswith("_"):  
                matched_keys = [k for k in transaction_dict if k.startswith(f)]
                matched_keys.sort()  
                state.extend([transaction_dict[k] for k in matched_keys])
            else:
                state.append(transaction_dict[f])
        return state


    def choose_action(self, transaction_dict, log=False):
        state = self.extract_state(transaction_dict)
        state_key = tuple(state)
        action = super().choose_action(state)
        if log:
            print(f"[TMA] Chose action {action} for state {state_key}, Q-values: {self.q_table.get(state_key)}")
        return action

    def update(self, transaction_dict, action, reward, next_transaction_dict, next_action=None):
        state = self.extract_state(transaction_dict)
        next_state = self.extract_state(next_transaction_dict) if next_transaction_dict is not None else None
        self.update_q_table(state, action, reward, next_state, next_action)

    def decay_epsilon(self, decay_rate=0.995, min_epsilon=0.01):
        self.epsilon = max(self.epsilon * decay_rate, min_epsilon)

