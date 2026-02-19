# Simulation Plan: Dual-Memory Cognitive Lock-in Model (V5)

Based on `docs/conceptual_model_v5.tex`. This document specifies the complete simulation scheme: implementation changes, experimental design, metrics, analysis, and execution plan.

---

## 1. Implementation Changes

### 1.1 New Module: `src/memory/normative.py`

```python
class NormativeMemory:
    """
    Rule-based normative memory. NOT FIFO.
    Stores a single internalised norm as a decision constraint.
    """

    # State
    norm: Optional[int]          # r_i ∈ {0, 1, None}  — which strategy is "the rule"
    strength: float              # σ_i ∈ [0, 1]         — how constraining
    anomaly_count: int           # a_i ∈ N              — accumulated violations
    evidence: float              # e_i ∈ R+             — DDM accumulator (pre-crystallisation)

    # Parameters
    ddm_noise: float             # σ_noise — Gaussian noise std in DDM (default: 0.1)
    crystal_threshold: float     # θ_crystal — evidence threshold for norm formation (default: 3.0)
    initial_strength: float      # σ_0 — norm strength at crystallisation (default: 0.8)
    crisis_threshold: int        # θ_crisis — anomalies before crisis (default: 10)
    crisis_decay: float          # λ_crisis — strength multiplier during crisis (default: 0.3)
    min_strength: float          # σ_min — below this, norm dissolves (default: 0.1)

    def update_evidence(self, trust: float, consistency: float,
                        signal_boost: float = 1.0) -> bool:
        """
        DDM evidence accumulation (Eq. 5-7 in v5.tex).
        Called each tick when norm is None.

        drift = (1 - trust) × consistency × signal_boost
        evidence = max(0, evidence + drift + N(0, σ²))

        Returns True if norm crystallised this tick.
        """

    def record_anomaly(self, observed_strategy: int) -> None:
        """
        Anomaly accumulation (Eq. 10 in v5.tex).
        Called when norm exists and observed strategy ≠ norm.
        Increments anomaly counter. Triggers crisis if threshold exceeded.
        """

    def check_crisis(self) -> bool:
        """
        Crisis check (Eq. 11 in v5.tex).
        If anomalies ≥ θ_crisis: strength *= λ_crisis, anomalies reset.
        If strength < σ_min: norm dissolved, evidence reset.
        Returns True if norm was dissolved.
        """

    def get_norm_belief(self) -> List[float]:
        """
        Returns one-hot belief vector if norm exists, else None.
        E.g., norm=0 → [1.0, 0.0]; norm=1 → [0.0, 1.0]; None → None
        """

    def has_norm(self) -> bool
    def get_state(self) -> dict  # for metrics collection
```

### 1.2 Modify: `src/agent.py`

**Add normative memory to Agent:**

```python
class Agent:
    def __init__(self, ..., normative_memory: Optional[NormativeMemory] = None):
        self._normative_memory = normative_memory

    def choose_action(self, use_signal=True) -> Tuple[int, int]:
        """
        Updated decision integration (Eq. 13-15 in v5.tex):

        b_exp = self._memory.get_strategy_distribution()

        if self._normative_memory and self._normative_memory.has_norm():
            compliance = self._normative_memory.strength ** compliance_exponent
            b_norm = self._normative_memory.get_norm_belief()
            b_eff = compliance * b_norm + (1 - compliance) * b_exp
        else:
            b_eff = b_exp

        return self._decision.choose_action(b_eff)
        """

    def process_observations(self, observations: List) -> None:
        """
        Dual-purpose observation processing (Section 6 in v5.tex):

        1. Feed to experience memory (existing: frequency extraction)
        2. Feed to normative memory:
           - If no norm: compute consistency → update DDM evidence
           - If has norm: check for violations → record anomalies
        """

    def get_enforcement_signal(self, observed_strategy: int) -> Optional[int]:
        """
        Violation-triggered enforcement (Eq. 16 in v5.tex):

        Conditions (ALL must hold):
        1. Has norm (r_i ≠ ∅)
        2. Norm strength σ_i > θ_enforce
        3. observed_strategy ≠ norm

        Returns: norm strategy to broadcast, or None
        """

    def receive_normative_signal(self, signaled_strategy: int) -> None:
        """
        Receiving normative signal boosts DDM drift (Eq. 17 in v5.tex):
        signal_boost = γ_signal (default 2.0)
        Applied on next update_evidence() call.
        """
```

**Update `create_agent()` factory:**

```python
def create_agent(agent_id, ...,
                 enable_normative: bool = False,
                 normative_params: dict = None) -> Agent:
    """Add normative_memory creation when enable_normative=True."""
```

### 1.3 Modify: `src/environment.py` or New: `src/environment_v5.py`

**Add enforcement phase to simulation loop:**

```python
def step(self) -> TickMetrics:
    """Updated tick sequence:

    1. Clear observations
    2. Random pair matching
    3. Execute interactions (action + prediction)
    4. Agent updates (experience memory + trust)
    5. Distribute observations (if observation_k > 0)
    6. Process observations → dual-purpose:
       a. Experience memory: add as data points
       b. Normative memory: DDM evidence OR anomaly tracking
    7. [NEW] Enforcement phase:
       - Each agent with norm checks recent observations for violations
       - If enforcement conditions met → broadcast normative signal
       - Receivers get signal_boost on next DDM update
    8. Collect metrics (including new normative metrics)
    9. Check convergence
    """
```

### 1.4 Modify: `src/norms/detector.py`

**Extend NormLevel from 5 to 6 levels:**

```python
class NormLevel(IntEnum):
    NONE = 0
    BEHAVIORAL = 1        # ≥95% adoption, stable 50 ticks
    EMPIRICAL = 2          # + belief error < 0.10  (renamed from COGNITIVE)
    SHARED = 3             # + belief variance < 0.05
    NORMATIVE = 4          # [NEW] ≥80% agents have internalised norm
    INSTITUTIONAL = 5      # + self-enforcing stability 200+ ticks
```

### 1.5 Modify: `config.py`

**Extend SimulationConfig:**

```python
@dataclass
class SimulationConfig:
    # ... existing fields ...

    # Normative memory parameters (new)
    enable_normative_memory: bool = False
    ddm_noise: float = 0.1
    crystal_threshold: float = 3.0
    initial_norm_strength: float = 0.8
    crisis_threshold: int = 10
    crisis_decay: float = 0.3
    min_norm_strength: float = 0.1

    # Enforcement parameters (new)
    enforce_threshold: float = 0.7
    compliance_exponent: float = 2.0
    signal_amplification: float = 2.0
```

### 1.6 New Metrics in `TickMetrics`

```python
@dataclass
class TickMetrics:
    # ... existing fields ...

    # Normative memory metrics (new)
    norm_adoption_rate: float       # fraction of agents with r_i ≠ ∅
    mean_norm_strength: float       # mean σ_i across agents with norms
    mean_ddm_evidence: float        # mean e_i across agents without norms
    total_anomalies: int            # sum of all anomaly counts
    norm_crises: int                # number of norms dissolved this tick
    enforcement_events: int         # number of normative signals broadcast this tick
    norm_crystallisations: int      # number of new norms formed this tick
```

---

## 2. Experimental Design

### 2.1 Core Experiment: 2×2 Factorial

The primary experiment isolates the contribution of each mechanism.

| Condition | Experience Memory | Normative Memory | Expected Max Level |
|-----------|:-:|:-:|---|
| **A: Baseline** | Fixed window (size=5) | OFF | BEHAVIORAL |
| **B: Lock-in Only** | Dynamic window [2,6] | OFF | SHARED |
| **C: Normative Only** | Fixed window (size=5) | ON | NORMATIVE (fragile) |
| **D: Full Model** | Dynamic window [2,6] | ON | INSTITUTIONAL |

**Fixed parameters across all conditions:**
- `N = 100`
- `T = 3000`
- `C0 = 0.5`
- `alpha = 0.1, beta = 0.3`
- `V = 0, Phi = 0.0`
- `n_trials = 50` per condition (200 total)
- Seeds: hash-based from base_seed=1000
- End condition: norm level 5 (institutional) or T reached

**Primary outcome measures:**
1. Convergence time (tick at which norm level 5 first reached; 0 if never)
2. Maximum norm level achieved
3. Final consensus strength (majority fraction at end)
4. Total ticks (simulation length)

**Expected results (hypothesised):**

| Metric | A: Baseline | B: Lock-in | C: Normative | D: Full |
|--------|:-:|:-:|:-:|:-:|
| Max norm level | 1 | 3 | 4 | 5 |
| Convergence (Level 5) rate | 0% | 0% | >0% | high |

### 2.2 Experiment 2: DDM Parameter Sensitivity

Sweep DDM parameters to characterise norm formation dynamics.

**2.2a: Crystallisation threshold θ_crystal**

| Parameter | Values |
|-----------|--------|
| `crystal_threshold` | [1.0, 2.0, 3.0, 5.0, 8.0] |
| Other params | Full model (Condition D) defaults |
| Trials | 30 per value |
| Seeds | 2000–2149 |

**Predictions:**
- Low θ → fast but potentially premature norm formation; might lock into wrong norm
- High θ → slow but robust norm formation; fewer wrong norms
- Optimal: θ ≈ 3.0 (balance speed vs accuracy)

**Measures:**
- Time to first crystallisation
- Fraction of agents with "correct" norm (matching final majority)
- Norm stability (does it survive 200+ ticks?)

**2.2b: DDM noise σ_noise**

| Parameter | Values |
|-----------|--------|
| `ddm_noise` | [0.01, 0.05, 0.1, 0.2, 0.5] |
| Other params | Full model defaults |
| Trials | 30 per value |
| Seeds | 2200–2349 |

**Predictions:**
- Low noise → deterministic, agents crystallise nearly simultaneously → fragile
- High noise → stochastic, wide spread in crystallisation time → heterogeneous population
- Optimal: σ ≈ 0.1 (natural individual differences without chaos)

**2.2c: Crisis threshold θ_crisis**

| Parameter | Values |
|-----------|--------|
| `crisis_threshold` | [3, 5, 10, 20, 50] |
| Other params | Full model defaults |
| Trials | 30 per value |
| Seeds | 2400–2549 |

**Predictions:**
- Low θ → norms are fragile, easily overthrown
- High θ → norms are rigid, hard to change even when wrong
- Tested with norm shock experiment (see 2.5)

### 2.3 Experiment 3: Trust Parameter Interaction

How trust asymmetry interacts with dual-memory mechanisms.

| Condition | α | β | Ratio β/α | Character |
|-----------|---|---|-----------|-----------|
| Symmetric-low | 0.05 | 0.05 | 1.0 | Slow, balanced |
| Symmetric-high | 0.2 | 0.2 | 1.0 | Fast, balanced |
| Asymmetric (default) | 0.1 | 0.3 | 3.0 | Slovic-inspired |
| Strong asymmetry | 0.05 | 0.4 | 8.0 | Very cautious |

**Cross with memory conditions:**
- Each trust setting × {Lock-in Only, Full Model} = 8 conditions
- 30 trials each = 240 runs
- Seeds: 3000–3239

**Key question:** Does trust asymmetry matter MORE or LESS when normative memory is present?

**Predictions:**
- In Lock-in Only: asymmetry critical (drives the window dynamics)
- In Full Model: asymmetry less critical (normative memory compensates)
- Strong asymmetry + Full Model: slowest norm formation (low drift) but most stable norms

### 2.4 Experiment 4: Observation Effect

Role of observations as the dual-purpose channel.

| Parameter | Values |
|-----------|--------|
| `observation_k` | [0, 1, 3, 5, 10] |
| Memory condition | Full model only |
| Trials | 30 per value |
| Seeds | 4000–4149 |

**Predictions:**
- k=0: No observations → norm formation impossible (no DDM input beyond own interactions)
- k=1: Minimal observations → slow norm formation, only agents with many interactions form norms
- k=3 (default): Balanced — enough signal for DDM without overwhelming experience memory
- k=10: Fast norm formation but potentially premature crystallisation
- Expected nonlinear threshold around k=2-3

**Measures:**
- DDM evidence trajectory (mean across agents)
- Time to first/50%/90% norm crystallisation
- Norm accuracy (fraction with correct norm)

### 2.5 Experiment 5: Norm Resilience (Shock Test)

Test H6: norm overthrow requires sustained anomalies, not gradual erosion.

**Protocol:**
1. Run Full Model until INSTITUTIONAL level reached (or tick 500, whichever first)
2. At shock_tick: introduce controlled violations
3. Measure recovery or collapse

**Shock types:**

| Shock | Description | Implementation |
|-------|-------------|----------------|
| **Mild** | 10% agents randomly switch strategy | Reset strategy, keep norm |
| **Moderate** | 30% agents switch + trust reset to 0.3 | Reset strategy + trust |
| **Severe** | 50% agents switch + trust reset + norm dissolved | Full reset for 50% |
| **Sustained** | 10% switch every 10 ticks for 100 ticks | Repeated perturbation |

- Full model only
- 30 trials per shock type = 120 runs
- Seeds: 5000–5119

**Predictions:**
- Mild: Fast recovery (<50 ticks), anomaly count stays below θ_crisis
- Moderate: Slower recovery (~100 ticks), some agents' norms may enter crisis
- Severe: May trigger norm collapse → re-crystallisation (norm can change)
- Sustained: Most revealing — tests anomaly accumulation mechanism directly

**Measures:**
- Recovery time (ticks to re-achieve convergence)
- Norm survival rate (fraction retaining original norm)
- Re-crystallisation events (new norms formed)
- Final norm = original norm? (path dependence)

### 2.6 Experiment 6: Population Scaling

Does the dual-memory model scale differently from experience-only?

| Parameter | Values |
|-----------|--------|
| `num_agents` | [20, 50, 100, 200, 500] |
| Memory condition | {Baseline, Lock-in Only, Full Model} |
| Trials | 20 per combination = 300 runs |
| Seeds | 6000–6299 |

**Predictions:**
- Baseline: convergence ~ O(N^0.7) (existing finding ~ O(N^0.6))
- Lock-in: convergence ~ O(N^0.6) (reproduces existing)
- Full Model: convergence ~ O(N^0.4) (enforcement creates cascade, sublinear)
- Enforcement should disproportionately help large populations

### 2.7 Experiment 7: Enforcement Threshold Sensitivity

When should agents enforce vs update?

| Parameter | Values |
|-----------|--------|
| `enforce_threshold` | [0.3, 0.5, 0.7, 0.9] |
| Memory condition | Full model |
| Trials | 30 per value |
| Seeds | 7000–7119 |

**Predictions:**
- θ=0.3: Too many enforcers too early → may enforce wrong norm
- θ=0.5: Moderate — some premature enforcement
- θ=0.7 (default): Only high-conviction (high-σ) agents enforce → robust
- θ=0.9: Very few enforcers → normative pressure weak, similar to Condition C

---

## 3. Metrics & Data Collection

### 3.1 Per-Tick Metrics (CSV columns)

**Existing (keep):**
```
tick, strategy_0_fraction, strategy_1_fraction,
mean_trust, coordination_rate, num_interactions,
strategy_switches, converged, majority_strategy,
majority_fraction, mean_memory_window,
norm_level, belief_error, belief_variance
```

**New (add):**
```
norm_adoption_rate,        # fraction of agents with r ≠ ∅
mean_norm_strength,        # mean σ across norm-holders
mean_ddm_evidence,         # mean e across non-norm-holders
total_anomalies,           # sum of all a_i
norm_crises,               # norms dissolved this tick
enforcement_events,        # normative signals sent this tick
norm_crystallisations,     # new norms formed this tick
correct_norm_rate,         # fraction with norm matching majority (post-hoc)
mean_compliance            # mean σ^k across norm-holders
```

### 3.2 Per-Agent Snapshot (at specific ticks)

Capture full agent state at ticks [0, 50, 100, 200, 500, 1000]:

```
agent_id, trust, memory_window, belief_0, belief_1,
has_norm, norm_strategy, norm_strength, anomaly_count,
ddm_evidence, current_strategy, total_enforcements
```

### 3.3 Event Log (optional, for detailed analysis)

Log individual events with columns:
```
tick, agent_id, event_type, details
```

Event types:
- `CRYSTALLISATION`: agent_id formed norm for strategy X at evidence level Y
- `CRISIS`: agent_id's norm dissolved after Z anomalies
- `ENFORCEMENT`: agent_id broadcast norm X after observing violation
- `SIGNAL_RECEIVED`: agent_id received normative signal for strategy X

---

## 4. Analysis Plan

### 4.1 Primary Analysis (2×2 Factorial)

**Statistical tests:**
- Two-way ANOVA: convergence_time ~ memory_type × normative_memory
- Expected interaction: dynamic + normative > additive effect
- Kruskal-Wallis for non-normal distributions (norm level is ordinal)
- Effect sizes: Cohen's d for pairwise comparisons

**Visualisations:**
1. **Strategy evolution curves** (4 panels, one per condition): mean ± SEM across trials
2. **Trust trajectory** (4 panels): mean trust over time
3. **Norm level progression** (4 panels): norm level over time (step function)
4. **Box plots**: convergence time by condition
5. **Norm adoption S-curve**: norm_adoption_rate over time for Conditions C & D
6. **Interaction plot**: convergence_time means with error bars, 2×2 layout

### 4.2 Hypothesis Testing

| Hypothesis | Test | Data | Expected |
|------------|------|------|----------|
| H1: Stochastic formation | Variance of crystallisation tick | Exp 1D | CV > 0.3 |
| H2: Low-trust first | Correlation: trust @ crystallisation vs order | Exp 1D | r < -0.5 |
| H3: Asymmetric response | Enforcement rate vs anomaly rate by norm-strength bin | Exp 1D | Significant split at θ_enforce |
| H4: Enforcement cascade | Norm adoption rate slope before/after first enforcement | Exp 1D | Slope increase > 2× |
| H5: Interaction effect | 2×2 ANOVA interaction term | Exp 1 | p < 0.01 |
| H6: Sustained anomaly | Norm survival under mild vs sustained shock | Exp 5 | Survival: mild > 90%, sustained < 50% |

### 4.3 DDM Dynamics Analysis (Experiment 2)

**Visualisations:**
1. **DDM evidence trajectories**: sample 10 agents, plot e_i(t) with crystallisation marked
2. **Crystallisation time histogram**: distribution across agents
3. **Phase portrait**: trust vs DDM evidence, with crystallisation boundary marked
4. **Parameter sensitivity heatmap**: convergence_time as function of (θ_crystal, σ_noise)

### 4.4 Cascade Analysis (Experiment 4 & 7)

Track the enforcement cascade:
1. Plot cumulative norm_adoption_rate with enforcement_events overlaid
2. Identify tipping point (Centola 2018): tick where adoption rate acceleration peaks
3. Measure cascade speed: ticks from 25% to 75% norm adoption
4. Compare tipping points across observation_k and enforce_threshold

---

## 5. Execution Plan

### Phase 1: Implementation (prerequisite)

| Task | File | Depends On |
|------|------|------------|
| 1a. Create NormativeMemory class | `src/memory/normative.py` | — |
| 1b. Update Agent with dual-memory | `src/agent.py` | 1a |
| 1c. Update create_agent factory | `src/agent.py` | 1a, 1b |
| 1d. Add enforcement phase to environment | `src/environment.py` | 1b |
| 1e. Extend NormLevel to 6 levels | `src/norms/detector.py` | — |
| 1f. Extend SimulationConfig | `config.py` | 1a |
| 1g. Extend TickMetrics | `src/environment.py` | 1b |
| 1h. Update experiment runner | `experiments/runner.py` | 1f |

**Validation after Phase 1:**
- Run single trial of each 2×2 condition with `--verbose`
- Confirm: Condition A produces no norms, Condition D produces norms
- Confirm: DDM evidence accumulates, crystallisation events occur
- Confirm: enforcement signals fire only under correct conditions

### Phase 2: Core Experiments

| Experiment | Runs | Est. Time* | Priority |
|------------|------|-----------|----------|
| Exp 1: 2×2 Factorial | 200 | ~20 min | **P0** |
| Exp 2a: θ_crystal sweep | 150 | ~15 min | P1 |
| Exp 2b: σ_noise sweep | 150 | ~15 min | P1 |
| Exp 2c: θ_crisis sweep | 150 | ~15 min | P1 |
| Exp 3: Trust × Memory | 240 | ~24 min | P1 |

*Estimated at ~6 sec/trial for 100 agents × 1000 ticks.

### Phase 3: Extension Experiments

| Experiment | Runs | Est. Time | Priority |
|------------|------|-----------|----------|
| Exp 4: Observation k | 150 | ~15 min | P2 |
| Exp 5: Shock resilience | 120 | ~15 min | P2 |
| Exp 6: Population scaling | 300 | ~60 min | P2 |
| Exp 7: Enforce threshold | 120 | ~12 min | P2 |

### Phase 4: Analysis & Visualisation

| Task | Input | Output |
|------|-------|--------|
| 4a. 2×2 factorial analysis | Exp 1 data | Figures 1-6, Table 1, ANOVA |
| 4b. DDM dynamics analysis | Exp 2 data | Figures 7-10, parameter heatmap |
| 4c. Cascade analysis | Exp 1D + Exp 4 + Exp 7 | Figures 11-13, tipping points |
| 4d. Hypothesis test table | All experiments | Table 2 (H1-H6 results) |
| 4e. Resilience analysis | Exp 5 data | Figures 14-15, recovery curves |
| 4f. Scaling analysis | Exp 6 data | Figure 16, scaling exponents |

---

## 6. Experiment Runner Configurations

### 6.1 Experiment 1: 2×2 Factorial

```python
conditions_2x2 = {
    "A_baseline": {
        "memory_type": "fixed",
        "fixed_memory_size": 5,
        "enable_normative_memory": False,
    },
    "B_lockin": {
        "memory_type": "dynamic",
        "dynamic_base_size": 2,
        "dynamic_max_size": 6,
        "enable_normative_memory": False,
    },
    "C_normative": {
        "memory_type": "fixed",
        "fixed_memory_size": 5,
        "enable_normative_memory": True,
    },
    "D_full": {
        "memory_type": "dynamic",
        "dynamic_base_size": 2,
        "dynamic_max_size": 6,
        "enable_normative_memory": True,
    },
}

shared_params = {
    "num_agents": 100,
    "max_ticks": 1000,
    "decision_mode": "cognitive_lockin",
    "initial_trust": 0.5,
    "alpha": 0.1,
    "beta": 0.3,
    "observation_k": 3,
    "convergence_threshold": 0.95,
    "convergence_window": 50,
    # Normative defaults (for C and D)
    "ddm_noise": 0.1,
    "crystal_threshold": 3.0,
    "initial_norm_strength": 0.8,
    "crisis_threshold": 10,
    "crisis_decay": 0.3,
    "enforce_threshold": 0.7,
    "compliance_exponent": 2.0,
    "signal_amplification": 2.0,
}

n_trials = 50
seeds = range(1000, 1000 + n_trials)
```

### 6.2 Experiment 2a: Crystallisation Threshold Sweep

```python
sweep_crystal = {
    "base_config": "D_full",  # Full model
    "sweep_param": "crystal_threshold",
    "values": [1.0, 2.0, 3.0, 5.0, 8.0],
    "n_trials": 30,
    "seeds": range(2000, 2150),
}
```

### 6.3 Experiment 5: Shock Resilience

```python
shock_protocol = {
    "base_config": "D_full",
    "run_until": "INSTITUTIONAL",  # or max 500 ticks
    "shock_types": [
        {"name": "mild",     "strategy_reset_frac": 0.1, "trust_reset": None,  "norm_reset": False},
        {"name": "moderate", "strategy_reset_frac": 0.3, "trust_reset": 0.3,   "norm_reset": False},
        {"name": "severe",   "strategy_reset_frac": 0.5, "trust_reset": 0.3,   "norm_reset": True},
        {"name": "sustained","strategy_reset_frac": 0.1, "trust_reset": None,  "norm_reset": False,
         "repeat_interval": 10, "repeat_count": 10},
    ],
    "post_shock_ticks": 500,
    "n_trials": 30,
    "seeds": range(5000, 5120),
}
```

---

## 7. File Output Structure

```
data/experiments/v5/
├── exp1_factorial/
│   ├── A_baseline/
│   │   ├── trial_1000_history.csv
│   │   ├── trial_1000_agents_t500.csv   # agent snapshot
│   │   └── trial_1000_events.csv        # event log
│   ├── B_lockin/
│   ├── C_normative/
│   ├── D_full/
│   ├── factorial_summary.csv            # one row per trial
│   └── factorial_results.json           # full config + results
├── exp2_ddm_sensitivity/
│   ├── crystal_sweep/
│   ├── noise_sweep/
│   └── crisis_sweep/
├── exp3_trust_interaction/
├── exp4_observation/
├── exp5_shock/
├── exp6_scaling/
├── exp7_enforcement/
└── figures/
    ├── fig1_strategy_evolution_2x2.pdf
    ├── fig2_trust_trajectory_2x2.pdf
    ├── fig3_norm_level_progression.pdf
    ├── fig4_convergence_boxplot.pdf
    ├── fig5_norm_adoption_scurve.pdf
    ├── fig6_interaction_plot.pdf
    ├── fig7_ddm_trajectories.pdf
    ├── fig8_crystal_histogram.pdf
    ├── fig9_phase_portrait.pdf
    ├── fig10_sensitivity_heatmap.pdf
    ├── fig11_cascade_timeline.pdf
    ├── fig12_tipping_point.pdf
    └── ...
```

---

## 8. Estimated Total Computation

| Experiment | Runs | Agents | Ticks | Total Agent-Ticks |
|------------|------|--------|-------|--------------------|
| Exp 1 | 200 | 100 | 1000 | 20M |
| Exp 2 (all) | 450 | 100 | 1000 | 45M |
| Exp 3 | 240 | 100 | 1000 | 24M |
| Exp 4 | 150 | 100 | 1000 | 15M |
| Exp 5 | 120 | 100 | 1500 | 18M |
| Exp 6 | 300 | ~175 avg | 1000 | 52.5M |
| Exp 7 | 120 | 100 | 1000 | 12M |
| **Total** | **1580** | | | **~186M** |

At ~6 sec/trial (100 agents): ~2.5 hours sequential, ~40 min with 4 workers.

---

## 9. Success Criteria

The simulation campaign is successful if:

1. **H5 confirmed**: 2×2 interaction effect is significant (p < 0.01), meaning dual-memory + lock-in > either alone
2. **Norm emergence observed**: Full model reaches NORMATIVE level in >80% of trials
3. **DDM mechanism validated**: crystallisation times show meaningful variance (H1) and correlate negatively with trust (H2)
4. **Enforcement cascade detected**: clear acceleration in norm adoption after first enforcement events (H4)
5. **Resilience asymmetry confirmed**: norms survive mild shocks but collapse under sustained anomalies (H6)
6. **Scaling advantage**: Full model shows lower scaling exponent than baseline

If H5 fails (no interaction effect), the dual-memory model does not add value beyond the individual mechanisms, and the theoretical framework needs revision.
