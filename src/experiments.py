import numpy as np
import matplotlib.pyplot as plt

from environment import ServerEnvironment
from agent import EpsilonGreedyAgent


# =========================================
# Run One Experiment
# =========================================

def run_experiment(epsilon, steps=1000):

    # Create server environment
    env = ServerEnvironment()

    # Create epsilon-greedy agent
    agent = EpsilonGreedyAgent(
        num_actions=env.num_servers,
        epsilon=epsilon
    )

    rewards = []
    actions = []

    # Run for 1000 steps
    for step in range(steps):

        # Agent chooses a server
        action = agent.select_action()

        # Environment gives reward
        reward, response_time = env.step(action)

        # Agent updates its Q-value
        agent.update(action, reward)

        # Save results
        rewards.append(reward)
        actions.append(action)

    return np.array(rewards), np.array(actions), agent.Q, agent.N


# =========================================
# Experiment Settings
# =========================================

epsilons = [0, 0.01, 0.1]

steps = 1000

runs = 100


# Store results
all_results = {}


# =========================================
# Run 100 Experiments for Each Epsilon
# =========================================

for epsilon in epsilons:

    all_rewards = []
    all_optimal = []

    for run in range(runs):

        rewards, actions, Q, N = run_experiment(
            epsilon,
            steps
        )

        all_rewards.append(rewards)

        # Server 4 is the optimal server
        optimal_actions = (actions == 3)

        all_optimal.append(optimal_actions)

    # Convert to NumPy arrays
    all_rewards = np.array(all_rewards)
    all_optimal = np.array(all_optimal)

    # Average results across 100 runs
    average_reward = np.mean(
        all_rewards,
        axis=0
    )

    optimal_percentage = np.mean(
        all_optimal,
        axis=0
    ) * 100

    # Save results
    all_results[epsilon] = {
        "rewards": average_reward,
        "optimal": optimal_percentage
    }


# =========================================
# Print Final Results
# =========================================

print()
print("========================================")
print("FINAL RESULTS")
print("========================================")

for epsilon in epsilons:

    rewards = all_results[epsilon]["rewards"]

    optimal = all_results[epsilon]["optimal"]

    print()
    print("Epsilon:", epsilon)

    print(
        "Average reward:",
        round(np.mean(rewards), 2)
    )

    print(
        "Final optimal-action percentage:",
        round(optimal[-1], 2),
        "%"
    )


# =========================================
# Create Results Folder
# =========================================

import os

os.makedirs(
    "results/figures",
    exist_ok=True
)


# =========================================
# Graph 1: Average Reward
# =========================================

plt.figure(figsize=(10, 6))

window = 50

for epsilon in epsilons:

    rewards = all_results[epsilon]["rewards"]

    # 50-step moving average
    moving_average = np.convolve(
        rewards,
        np.ones(window) / window,
        mode="valid"
    )

    plt.plot(
        moving_average,
        label=f"epsilon = {epsilon}"
    )

plt.xlabel("Steps")

plt.ylabel("Average Reward")

plt.title("Average Reward over Time")

plt.legend()

plt.grid()

plt.tight_layout()

plt.savefig(
    "results/figures/average_reward.png",
    dpi=300
)

# IMPORTANT:
# Do NOT use plt.show()
plt.close()


# =========================================
# Graph 2: Optimal Server Selection
# =========================================

plt.figure(figsize=(10, 6))

for epsilon in epsilons:

    optimal = all_results[epsilon]["optimal"]

    plt.plot(
        optimal,
        label=f"epsilon = {epsilon}"
    )

plt.xlabel("Steps")

plt.ylabel("Optimal Server Selection (%)")

plt.title("Optimal Server Selection over Time")

plt.legend()

plt.grid()

plt.tight_layout()

plt.savefig(
    "results/figures/optimal_action.png",
    dpi=300
)

# IMPORTANT:
# Do NOT use plt.show()
plt.close()


# =========================================
# Finished
# =========================================

print()
print("========================================")
print("EXPERIMENT COMPLETED SUCCESSFULLY")
print("========================================")

print()
print("Graphs saved in:")

print("results/figures/average_reward.png")

print("results/figures/optimal_action.png")