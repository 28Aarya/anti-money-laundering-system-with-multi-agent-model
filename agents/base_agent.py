import numpy as np
import random

class BaseAgent:
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=0.1, algorithm='Q-learning'):
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.algorithm = algorithm  # Either Q-learning or SARSA
        self.q_table = {}  # Q-table to store Q-values

    def choose_action(self, state):
        """ Choose action based on the current state using epsilon-greedy policy. """
        state_key = tuple(round(x, 3) for x in state)
        if self.algorithm == 'Q-learning':
            return self._q_learning_action(state_key)
        elif self.algorithm == 'SARSA':
            return self._sarsa_action(state_key)
        else:
            raise ValueError(f"Invalid algorithm chosen: {self.algorithm}")

    def _q_learning_action(self, state_key):
        """ Choose action using Q-learning's epsilon-greedy policy. """
        if state_key not in self.q_table:
            self.q_table[state_key] = [0, 0]  # Initialize with zero for both actions

        if random.uniform(0, 1) < self.epsilon:
            return random.choice([0, 1])  # Explore
        else:
            return np.argmax(self.q_table[state_key])  # Exploit (choose the best action)

    def _sarsa_action(self, state_key):
        """ SARSA action selection - choose next action based on current policy. """
        if state_key not in self.q_table:
            self.q_table[state_key] = [0, 0]  # Initialize with zero for both actions

        if random.uniform(0, 1) < self.epsilon:
            return random.choice([0, 1])  # Explore
        else:
            return np.argmax(self.q_table[state_key])  # Exploit (choose the best action)

    def update_q_table(self, state, action, reward, next_state, next_action):
        """ Update Q-values based on Q-learning or SARSA formula. """
        state_key = tuple(round(x, 3) for x in state)
        if next_state is None:
            # Terminal state: only update with immediate reward, no future value
            if state_key not in self.q_table:
                self.q_table[state_key] = [0, 0]
            old_q = self.q_table[state_key][action]
            self.q_table[state_key][action] = old_q + self.alpha * (reward - old_q)
            return
        next_state_key = tuple(round(x, 3) for x in next_state)
        if state_key not in self.q_table:
            self.q_table[state_key] = [0, 0]
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = [0, 0]

        old_q = self.q_table[state_key][action]
        if self.algorithm == 'Q-learning':
            # Q-learning update rule
            best_next_action = np.argmax(self.q_table[next_state_key])
            self.q_table[state_key][action] = old_q + self.alpha * (reward + self.gamma * self.q_table[next_state_key][best_next_action] - old_q)
        elif self.algorithm == 'SARSA':
            # SARSA update rule
            self.q_table[state_key][action] = old_q + self.alpha * (reward + self.gamma * self.q_table[next_state_key][next_action] - old_q)
        else:
            raise ValueError(f"Invalid algorithm chosen: {self.algorithm}")
