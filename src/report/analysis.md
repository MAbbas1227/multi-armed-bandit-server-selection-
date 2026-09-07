# Multi-Armed Bandit Analysis — Scenario 6: Server Selection for Web Requests

## 1. Scenario Description

We model a cloud load balancer as a **stationary multi-armed bandit** problem. There are 5 arms
(Server 1–5), each representing a routing choice for an incoming request. Selecting a server
produces a random response time drawn from that server's hidden distribution.

Since lower response time is better, but bandit algorithms are built to *maximize* reward, we
define:

```
reward R = -response_time
```

A fast, low-response-time server therefore yields a reward close to 0, while a slow server yields
a very negative reward. The load balancer's goal is to learn — purely from observed rewards, with
no access to the underlying distributions — which server to route to most often.

## 2. Key Definitions

- **q\*(a)** — the true, fixed average reward of server *a*, determined by its hidden mean
  response time. This value is set once when the environment is created and is **never visible**
  to the learning agent.
- **Q_t(a)** — the agent's running *estimate* of q\*(a) at time step t, built only from the
  rewards actually observed so far. It starts at 0 for every server and is updated every time
  that server is selected.
- **N(a)** — a counter of how many times server *a* has been selected. It is the denominator in
  the incremental update rule below.

## 3. ε-Greedy Action Selection

With probability ε, the agent ignores its current estimates and picks a server **uniformly at
random** (exploration). With probability 1−ε, it picks whichever server currently has the
highest Q(a), breaking ties randomly (exploitation).

- **ε = 0** — pure exploitation. The agent locks onto whatever looked best early and never
  deliberately deviates from it.
- **ε = 0.01** — almost always exploits, but roughly 1 in 100 steps is a forced random pick.
- **ε = 0.1** — exploits 90% of the time, but a full 10% of all 1000 steps are forced random
  picks, regardless of how confident the current estimates are.

This is visible directly in our results: at ε = 0.1, even Server 3 — the **worst** server
(q\*(3) = −120) — was still selected 18 times, purely from forced exploration. That would not
happen under ε = 0.

## 4. Incremental Update Rule — Worked Example

The agent updates its estimate using the incremental sample-average rule:

```
Q_new(a) = Q_old(a) + (1 / N(a)) * (R - Q_old(a))
```

**Example:** Suppose Server 2 currently has Q_old = −72.00 after 9 prior selections, and it is
now selected for the 10th time (N = 10), observing a reward of R = −68:

```
Q_new = -72 + (1/10) * (-68 - (-72))
      = -72 + (1/10) * (4)
      = -72 + 0.4
      = -71.6
```

Each new sample nudges the estimate closer to the true mean, and the step size (1/N) shrinks as
N grows — this is what makes it a genuine running average rather than a fixed learning rate.

## 5. Experimental Setup

- **Servers (arms):** 5
- **Hidden true means (response time, ms):** [100, 70, 120, 50, 85] → q\*(a) = [−100, −70, −120, −50, −85]
- **Optimal server:** Server 4 (q\*(4) = −50, lowest response time)
- **Steps per run:** 1000
- **ε values compared:** 0, 0.01, 0.1

## 6. Results Table (from our run)

| Server | q\*(a) (true) | Q(a), ε=0 | Q(a), ε=0.01 | Q(a), ε=0.1 | N(a), ε=0.1 |
|--------|--------------:|----------:|-------------:|------------:|------------:|
| 1      | −100          | −77.64    | −102.54      | −109.52     | 15          |
| 2      | −70           | −95.54    | −73.06       | −70.50      | 17          |
| 3      | −120          | −130.32   | −125.67      | −111.42     | 18          |
| **4**  | **−50**       | **−50.00**| **−50.07**   | **−50.24**  | **925**     |
| 5      | −85           | −89.28    | −81.97       | −85.60      | 25          |

**Overall performance:**

| ε     | Average reward | Final optimal-action % |
|-------|----------------:|------------------------:|
| 0     | −50.21          | 100.0%                  |
| 0.01  | −50.58          | 99.0%                   |
| 0.1   | −53.69          | 97.0%                   |

## 7. Interpreting the Estimation Accuracy

At ε = 0, Q(4) converged to −50.00 — essentially exact. But this is misleading: the agent only
achieved this because it happened to try Server 4 early and never revisited the others (N=1 for
three of the five servers). Its estimates for the non-optimal servers are wildly inaccurate
(e.g. Q(2) = −95.54 vs. the true −70).

By contrast, ε = 0.01 and ε = 0.1 produce progressively better-calibrated estimates **across all
five servers**, not just the optimal one, because they kept sampling the "losing" arms
throughout the run. This is an important and slightly counter-intuitive result: **ε = 0 gave the
best final reward and 100% optimal selection, but the least accurate overall model of the
environment.**

## 8. Graph Interpretation

**`average_reward.png`:** All three ε values converge within the first ~20 steps, then hold
roughly flat for the rest of the run. ε = 0 sits highest, around −50. ε = 0.01 tracks just below
it, around −50.5. ε = 0.1 is clearly separated and worse, oscillating between −53 and −54 for the
entire run. The gap between ε = 0.1 and the other two never closes — it is a genuine, ongoing cost
of continued exploration, paid on every step for the life of the run.

**`optimal_action.png`:** All three ε values shoot up from roughly 15% to 90%+ within the first
20–30 steps. ε = 0 and ε = 0.01 converge to ~99–100% and stay essentially flat there. ε = 0.1
plateaus lower, around 90–97%, and stays visibly noisier throughout — because 10% of every step is
a forced random pick, which caps its best possible long-run optimal-selection rate at roughly
90% + (0.1 × 1/5) ≈ 92%, consistent with what the graph shows.

## 9. Why Different ε Values Produce Different Behavior

Higher ε trades short-term reward for continued information-gathering. In this task, the five
servers' true means are far enough apart (50 to 120) relative to their standard deviations
(8 to 20) that the agent receives a very clear, low-noise signal about which server is best,
almost immediately. Continued exploration under ε = 0.1 therefore has little upside and a real,
measurable cost in this specific environment.

This explains why ε = 0 performed best here: pure-greedy got lucky on which arm it sampled well
early, and because rewards are cleanly separated between servers, that early signal was reliable
rather than misleading. In a noisier or **non-stationary** environment — for example, if server
response times drifted over time — ε = 0 would be far riskier, since it would never re-check
whether the previously-best server was still actually best. This is worth flagging explicitly,
since the assignment notes we may be asked to adapt this implementation to a non-stationary
setting during evaluation.

## 10. Findings

- All three ε values successfully identify Server 4 as optimal, converging within roughly the
  first 2–3% of the run.
- ε = 0 achieves the highest average reward and optimal-selection percentage, but at the cost of
  a poorly-estimated model of the non-optimal servers.
- ε = 0.01 offers a close approximation of ε = 0's performance while retaining a small amount of
  ongoing verification of the other servers.
- ε = 0.1 sacrifices roughly 3–4 units of average reward per step compared to ε = 0, in exchange
  for a much more complete and accurate picture of every server's true performance.
- The clean separation between the servers' true means (relative to their variance) is the main
  reason greedy exploitation performs so well in this particular, stationary setting.

## 11. Conclusion

For this stationary, low-noise 5-server bandit, all three ε values learn to identify the optimal
server quickly, but they trade off differently: ε = 0 maximizes short-run reward and
optimal-selection rate but leaves its model of the non-optimal servers highly inaccurate and
undertested; ε = 0.1 sacrifices some average reward in exchange for a much more complete and
accurate picture of every server's true performance; ε = 0.01 sits close to the ε = 0 outcome
while retaining a small amount of continued verification. The right choice in practice depends on
whether the environment is expected to stay stationary (favoring low ε) or might drift over time
(favoring a nonzero ε for continued monitoring).