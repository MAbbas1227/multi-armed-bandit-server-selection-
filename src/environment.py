import numpy as np


class ServerEnvironment:

    def __init__(self):
        # Hidden true server parameters
        self.means = np.array([100, 70, 120, 50, 85])
        self.std = np.array([15, 10, 20, 8, 12])

        # True reward values q*(a)
        self.true_values = -self.means

        # Number of servers
        self.num_servers = 5

    def step(self, action):
        # Generate a random response time
        response_time = np.random.normal(
            self.means[action],
            self.std[action]
        )

        # Convert response time into reward
        reward = -response_time

        return reward, response_time