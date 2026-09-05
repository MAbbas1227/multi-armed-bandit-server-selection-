# multi-armed-bandit-server-selection-


## 📌 Project Overview

This project models **cloud server selection as a Stationary Multi-Armed Bandit (MAB) problem**.

A cloud application receives multiple web requests and must choose one of five available servers. Each server has a different response-time distribution due to differences in hardware, workload, and performance.

The goal of the agent is to learn which server provides the best performance while balancing:

- **Exploration** — trying different servers to learn about them
- **Exploitation** — selecting the server currently estimated to be the best

We use the **ε-greedy algorithm** and compare three exploration rates:

- ε = 0
- ε = 0.01
- ε = 0.1

The project uses an **incremental sample-average method** to estimate the value of each server.

---

## 🎯 Problem Statement

The system has five servers:

- Server 1
- Server 2
- Server 3
- Server 4
- Server 5

The true response-time distributions of the servers are hidden from the agent.

For every request:

1. The agent selects a server.
2. The selected server produces a stochastic response time.
3. The response time is converted into a reward.
4. The agent updates its estimated value of that server.
5. The agent continues learning over many requests.

Since lower response time is better, we define reward as:

```text
Reward = -Response Time
