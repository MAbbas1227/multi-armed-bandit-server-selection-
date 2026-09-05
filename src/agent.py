import numpy as np


class EpsilonGreedyAgent:

    def __init__(self, num_actions, epsilon):
        self.num_actions = num_actions
        self.epsilon = epsilon

        # Estimated action values Q(a)
        self.Q = np.zeros(num_actions)

        # Number of times each action was selected
        self.N = np.zeros(num_actions, dtype=int)

    def select_action(self):

        # Explore
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)

        # Exploit
        best_actions = np.flatnonzero(self.Q == np.max(self.Q))
        return np.random.choice(best_actions)

    def update(self, action, reward):

        # Increase action count
        self.N[action] += 1

        # Incremental sample-average update
        self.Q[action] = self.Q[action] + (
            1 / self.N[action]
        ) * (reward - self.Q[action])