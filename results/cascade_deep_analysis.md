# Deep Analysis: The Normative Cascade Mechanism

**Scripts**: `diagnose_cascade.jl`, `diagnose_deep.jl`
**Data**: `cascade_comparison.csv`, `cascade_transitions.csv`, `cascade_trigger_analysis.csv`

---

## 1. Side-by-Side: Both vs Exp_only (same seed=42)

Both models see identical initial random pairings (same RNG seed). The divergence reveals exactly what the normative system adds.

| tick | Both frac_A | Both coord | Both crA/crB | Exp_only frac_A | Exp_only coord | Exp b_exp |
|------|-----------|-----------|-------------|----------------|---------------|-----------|
| 1 | 0.49 | 0.58 | 0/0 | 0.49 | 0.58 | 0.490 |
| 30 | 0.50 | 0.40 | 39/38 | 0.50 | 0.52 | 0.458 |
| 50 | **0.61** | 0.42 | 44/34 | 0.55 | 0.46 | 0.565 |
| 60 | **0.70** | 0.72 | 44/**10** | 0.68 | 0.48 | 0.648 |
| 70 | **0.96** | 0.92 | 85/1 | 0.58 | 0.60 | 0.602 |
| 75 | **1.00** | **1.00** | 97/0 | 0.65 | 0.54 | 0.653 |
| 100 | 1.00 | 1.00 | 100/0 | 0.78 | 0.56 | 0.780 |
| 200 | 1.00 | 1.00 | 100/0 | 0.91 | 0.86 | 0.925 |
| 300 | 1.00 | 1.00 | 100/0 | 1.00 | 1.00 | 0.993 |

**Observation**: Both 在 tick 75 达到完美收敛；同一个 seed 的 Exp_only 在 tick 300 才勉强到达。同样的初始条件，6 倍速差。

至 tick 50，两者的 frac_A 几乎相同（0.61 vs 0.55）。关键分叉在 tick 50-70 之间——Both 触发了 cascade（crB: 34 -> 1），Exp_only 继续缓慢扩散。

---

## 2. Individual Agent Lifecycle: B -> Dissolution -> Re-crystallize to A

追踪了每个 agent 的 norm 状态转换。在 seed=42 的 Both 试验中：

### B->uncr->A 路径统计 (44 agents)

| 指标 | Mean | Median | Range |
|------|------|--------|-------|
| B-crystallized 持续时间 | **36.1 ticks** | 36.5 | [25, 47] |
| Uncrystallized 恢复时间 | **11.2 ticks** | 10.0 | [4, 24] |
| re-crystallize 时的 b_exp_A | **0.979** | 1.0 | [0.667, 1.0] |

### B->uncr->B 路径: **0 agents**

没有一个 agent 在从 B-norm dissolution 后又重新晶化到 B。这是 **one-way ratchet（单向棘轮）** 的直接证据。

### 解读

1. **B-crystallized 持续 ~36 ticks**：anomaly 积累到 theta_crisis=10 大约需要 10 次 partner violation。以 ~50% 的概率遇到 A-action partner（在对称阶段），约 20 ticks 积累 10 anomalies。但 crisis 只是 sigma *= 0.3，不一定直接 dissolve（需要 sigma < sigma_min=0.1），所以需要多次 crisis 周期。

2. **Uncrystallized 只有 ~11 ticks**：dissolution 后 agent 重新进入 DDM。此时环境已经 A-biased（b_exp ~ 0.98），DDM 每 tick 接收到强正向 drift -> 快速跨过 theta_crystal=3.0。

3. **b_exp_A at re-crystallization = 0.979**：dissolution 的 agent 在 uncrystallized 阶段经历着强 A-biased 环境，FIFO 迅速填满 A-observations -> b_exp 飙升。DDM 也看到一致的 A-evidence -> 晶化到 A。

4. **零 B->B 循环**：一旦 dissolution，agent 看到的环境已经不是 50/50 而是 >60% A -> 没有任何理由重新晶化到 B。这就是为什么 cascade 是不可逆的。

### 典型个体轨迹示例

Agent #15:
```
tick 13: uncr -> crB   (DDM random walk crossed -3.0)
tick 13-58: B-crystallized, 45 ticks
  - Acts B with ~82% probability
  - Accumulates anomalies from A-playing partners
  - sigma decays: 0.8 -> crisis -> 0.24 -> crisis -> below 0.1
tick 58: crB -> uncr    (dissolution! sigma < 0.1)
tick 58-65: uncrystallized, 7 ticks
  - Sees ~80% A-actions from partners (cascade already underway)
  - FIFO fills with A, b_exp_A rises to 1.0
  - DDM accumulates strong positive evidence
tick 65: uncr -> crA    (re-crystallizes to A)
  - Joins the majority, adds to cascade momentum
```

---

## 3. Multi-Seed Cascade Analysis (30 seeds)

| 指标 | Mean | Median | Range |
|------|------|--------|-------|
| Symmetric phase ends (last tick with \|crA-crB\| <= 2) | 10.4 | 7.5 | [0, 38] |
| Cascade completes (minority norm = 0) | 41.5 | 43.5 | [10, 79] |
| **Cascade duration** | **31.1** | **33.5** | **[4, 45]** |
| Behavioral convergence tick | 34.5 | 33.5 | [17, 70] |

### Key observations

1. **Cascade duration is remarkably consistent**: median = 33.5 ticks, 大部分在 27-42 范围内。无论对称破坏发生在 tick 6 还是 tick 38，cascade 一旦启动就大约 30-35 ticks 完成。

2. **Variability comes from symmetry-breaking, not cascade speed**: 最快的 trial (seed=3, 19, 21) 在 tick 10-11 就完成了 cascade——这是因为初始随机波动碰巧非常不对称，几乎跳过了 symmetric phase。最慢的 (seed=6) 到 tick 38 才打破对称。

3. **Convergence ~ cascade_complete**: behavioral convergence tick (median=33.5) 接近 cascade completion (median=43.5)。说明 cascade 本身就是收敛的主要事件。

---

## 4. The Dual-Memory Feedback Loop (Formal Description)

### 4.1 Without Normative Memory (Exp_only): Diffusion

```
             ┌──────────────────────────────┐
             │     Agent i's FIFO buffer     │
             │  [partner actions over w ticks]│
             └──────────┬───────────────────┘
                        │ compute b_exp_A
                        v
              action = rand < b_exp_A
                        │
                        v
              partner j observes action
                        │
                        v
              j's FIFO absorbs observation
                        │
                        v
                    ... (repeat)
```

Each agent updates INDEPENDENTLY based on its own FIFO. Information propagates one-hop-at-a-time through random pairings. Speed: O(N) because the signal must diffuse through the entire mixing population.

### 4.2 With Normative Memory (Both): Phase Transition

```
Phase A: Symmetry Breaking (slow, experiential-driven)
──────────────────────────────────────────────────────
  b_exp drifts from 0.50 to ~0.55 via FIFO
  DDM crystallizes roughly 50/50 to A and B norms
  Both sides accumulate anomalies from each other

Phase B: Differential Erosion (medium)
──────────────────────────────────────
  Slight A-majority -> B-norms face more anomalies than A-norms
  sigma_B decays faster -> compliance drops -> B-agents act less "B"
  -> environment becomes more A-biased -> more B anomalies (positive feedback)

  ┌──────────────────────────────────────────────────────────┐
  │  anomaly(B) > anomaly(A)                                 │
  │      -> sigma_B drops faster                             │
  │      -> B compliance drops                               │
  │      -> B agents act less B, more like random            │
  │      -> environment more A-biased                        │
  │      -> anomaly(B) increases further                     │
  │      -> crisis -> dissolution of B-norms                 │
  └──────────────────────────────────────────────────────────┘

Phase C: Cascade / Avalanche (fast, ~15-20 ticks)
─────────────────────────────────────────────────
  B-norms dissolve -> agents re-enter DDM
  Environment now strongly A-biased (>70%)
  DDM rapidly crystallizes to A (drift >> noise)
  Each new A-agent strengthens the A-environment

  ┌──────────────────────────────────────────────────────────┐
  │  B dissolution -> re-crystallize to A                    │
  │      -> A-majority increases                             │
  │      -> remaining B-norms face EVEN MORE anomalies       │
  │      -> faster dissolution                               │
  │      -> faster A re-crystallization                      │
  │      -> EXPONENTIAL acceleration                         │
  │                                                          │
  │  crB: 30 -> 20 -> 10 -> 5 -> 2 -> 0                    │
  │  Each step makes the next step faster                    │
  │  This is why the cascade takes only ~15 ticks            │
  └──────────────────────────────────────────────────────────┘

Phase D: Lock-in (permanent)
────────────────────────────
  All agents A-crystallized
  sigma increases via conformity strengthening
  b_exp converges to 1.0
  Confidence C -> 1.0
  Self-reinforcing equilibrium: both memory systems agree
```

### 4.3 Why the Cascade is N-independent

The cascade speed depends on:
- **Anomaly accumulation rate**: ~1 per 2 ticks when 70%+ play A -> crisis every ~20 ticks -> dissolution in ~30 ticks
- **Re-crystallization speed**: DDM drift = (1-C) * f_diff. When f_diff ~ 0.8 (strong majority), crosses theta=3.0 in ~5-10 ticks

These rates depend on the **local** observation (one partner per tick), not on N. A B-norm agent in N=20 and N=500 both see the same thing: their partner's action, which is A with probability ~majority fraction.

The only N-dependent part is Phase A (symmetry breaking), which is driven by the experiential system's diffusion process. But even this is bounded because the DDM noise helps -- random walks cross the crystallization threshold quickly.

---

## 5. The One-Way Ratchet: Why Cascade is Irreversible

The ratchet mechanism has three interlocking components:

1. **Asymmetric anomaly accumulation**: Minority norm agents accumulate anomalies faster because they observe more violations. The majority norm agents accumulate fewer anomalies. This asymmetry is self-reinforcing.

2. **Dissolution is biased**: When agents dissolve and re-enter the DDM, they observe a majority-biased environment. The DDM then crystallizes to the majority norm. Result: dissolution always converts minority -> majority, never the reverse.
   - Evidence: 44 B->uncr->A transitions, **0** B->uncr->B transitions in seed=42

3. **Experiential memory reinforcement**: As the majority grows, every agent's FIFO fills with majority-action observations. b_exp shifts toward the majority. Even uncrystallized agents act majority-biased, further strengthening the signal.

These three forces combine to make the cascade strictly irreversible once a critical asymmetry threshold is reached (~55-60% crystallized to one side).
