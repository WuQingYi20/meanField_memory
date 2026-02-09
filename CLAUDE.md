# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an agent-based model (ABM) simulation studying how **memory mechanisms affect norm formation** in coordination games. Agents play pure coordination games with adaptive memory systems, where prediction accuracy drives trust dynamics, which in turn modulates memory capacity.

The key innovation is the **feedback loop**: `Prediction Accuracy → Trust → Memory Window → Beliefs → Actions → Prediction Accuracy`

## Running the Simulation

### Basic Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run default simulation (100 agents, cognitive_lockin mode, fixed memory)
python main.py

# Quick test (10 agents, 100 ticks)
python main.py --quick

# Run with specific decision mode
python main.py --mode cognitive_lockin    # Recommended: probability matching with trust
python main.py --mode dual_feedback       # Original τ-based softmax
python main.py --mode epsilon_greedy      # Best response with exploration

# Run with different memory types
python main.py --memory fixed             # Fixed window (default)
python main.py --memory decay --decay-rate 0.8
python main.py --memory dynamic --dynamic-max 6  # Trust-linked adaptive window

# Run comparison experiments
python main.py --experiment               # Compare all memory types
python experiments/runner.py              # Decision mode comparison
python experiments/runner.py --trust      # Trust parameter comparison

# Launch interactive dashboard
python main.py --dashboard
```

### Common Development Tasks

```bash
# Run a single test simulation
python main.py --agents 50 --ticks 500 --seed 123 --verbose

# Compare memory types with 10 trials each
python -c "from experiments.runner import run_memory_comparison; run_memory_comparison(num_agents=100, n_trials=10)"

# Compare decision modes
python -c "from experiments.runner import run_decision_mode_comparison; run_decision_mode_comparison(n_trials=10)"

# Run trust parameter sweep
python -c "from experiments.runner import run_trust_parameter_comparison; run_trust_parameter_comparison(n_trials=10)"

# Full factorial comparison (memory × decision mode)
python -c "from experiments.runner import run_full_comparison; run_full_comparison(n_trials=10)"

# Agent count scaling analysis
python -c "from experiments.runner import run_agent_count_sweep; run_agent_count_sweep(agent_counts=[10, 50, 100, 200])"
```

## Architecture

### Core System Components

The simulation has **three layered subsystems** that interact:

1. **Memory System** (`src/memory/`)
   - `BaseMemory`: Abstract interface for all memory types
   - `FixedMemory`: Sliding window with equal weights
   - `DecayMemory`: Exponentially weighted by recency (λ^age)
   - `DynamicMemory`: **Trust-linked adaptive window** [2, 6] (key innovation)
   - Each memory computes weighted strategy distribution from interaction history

2. **Decision System** (`src/decision/`)
   - `BaseDecision`: Abstract interface for decision mechanisms
   - `CognitiveLockInDecision`: **Recommended** - Probability matching where trust updates asymmetrically (α=0.1, β=0.3 per Slovic 1993)
   - `DualFeedbackDecision`: Original model with τ-based softmax (two feedback loops)
   - `EpsilonGreedyDecision`: Best response + exploration (robustness check)
   - Decision mechanisms choose actions AND make predictions (prediction error drives trust updates)

3. **Agent Integration** (`src/agent.py`)
   - Integrates memory + decision + trust
   - **Critical linkage**: For `DynamicMemory`, the memory window queries decision's `get_trust()` to compute effective window size
   - Supports observation-based learning, normative expectations (Bicchieri 2006), and pre-play signaling (Skyrms 2010)

### Key Architectural Patterns

**Factory Pattern for Agent Creation**: Use `create_agent()` in `src/agent.py` to instantiate agents with appropriate memory/decision combinations. This handles the complex wiring between trust and memory.

**Strategy Pattern for Interchangeable Mechanisms**: Both memory and decision use abstract base classes, enabling clean comparison experiments.

**Trust-Memory Coupling**: The `_link_memory_to_trust()` method in `Agent.__init__()` creates a lambda that lets `DynamicMemory` access the decision mechanism's trust value. This is the **core feedback loop**.

### Decision Modes Explained

The codebase supports **three distinct decision modes** for comparative research:

- **`cognitive_lockin`** (RECOMMENDED): Agents use probability matching (select action proportional to beliefs) and trust updates via prediction accuracy. Trust is "hard to build, easy to break" (α < β). This creates **cognitive lock-in**: as agents align, trust increases, memory lengthens, beliefs stabilize, reinforcing consensus. **Fastest convergence** in experiments.

- **`dual_feedback`**: Original model where temperature τ drives both action selection (softmax) AND memory (via trust). Has two feedback loops: behavioral (τ affects choice randomness) and cognitive (τ affects memory via trust). Requires ~75% prediction accuracy to maintain τ=1.0, which can prevent convergence from 50-50 initial states.

- **`epsilon_greedy`**: Best response with ε-greedy exploration (ε = 1 - trust). Used for robustness checks to verify results aren't artifacts of probability matching.

### Environment and Simulation Flow

`SimulationEnvironment` (`src/environment.py`) orchestrates:
1. Random pairing of agents each tick
2. Simultaneous action selection + prediction
3. Game execution (coordination payoffs)
4. Agent updates (memory + trust)
5. Optional observation distribution (if `observation_k > 0`)
6. Norm detection using Bicchieri (2006) framework
7. Convergence tracking (95% majority for 50 ticks)

The simulation supports **observation-based learning**: agents can observe `k` random interactions beyond their own, modeling social learning.

### Configuration System

`config.py` defines:
- `SimulationConfig`: Full parameter set for a single run
- Preset configs: `COGNITIVE_LOCKIN_CONFIG`, `DUAL_FEEDBACK_CONFIG`, etc.
- Trust parameter presets: asymmetric (default), symmetric (fast), sensitive (high responsiveness)

`experiments/runner.py` uses `ExperimentConfig` for batch experiments with parallel execution support.

## Important Implementation Details

### Trust Dynamics

Trust updates occur in decision mechanism's `update()` method based on prediction accuracy:
- **CognitiveLockIn/EpsilonGreedy**: `T_new = T + α if correct else T - β` (clamped to [0,1])
- **DualFeedback**: `τ_new = τ × (1 - cooling_rate) if correct else τ + heating_penalty` (clamped to [τ_min, τ_max]), then `trust = 1 - (τ - τ_min) / (τ_max - τ_min)`

The **asymmetry** (α < β) creates negativity bias: trust breaks faster than it builds (Slovic 1993).

### Memory Window Calculation

For `DynamicMemory`, the effective window is:
```python
window = base_size + floor(trust * (max_size - base_size))
# Default: base_size=2, max_size=6 → range [2, 6]
```

At trust=0.5 (initial): window ≈ 4. At trust=1.0: window = 6. At trust=0: window = 2.

### Norm Detection

`NormDetector` (`src/norms/detector.py`) implements Bicchieri (2006) multi-level framework:
- **Level 0 (NONE)**: No regularity
- **Level 1 (BEHAVIORAL)**: Behavioral regularity (≥95% adoption)
- **Level 2 (COGNITIVE)**: + Accurate beliefs (low belief error)
- **Level 3 (SHARED)**: + Belief consensus (low belief variance)
- **Level 4 (INSTITUTIONAL)**: + Stability over time

### Coordination Game

`CoordinationGame` (`src/game.py`): Pure coordination - payoff = 1 if strategies match, 0 otherwise. Agents observe partner's strategy anonymously (no identity tracking).

### Visualization

- `visualization/static_plots.py`: Publication-quality matplotlib figures (strategy evolution, trust, convergence)
- `visualization/dashboard.py`: Streamlit interactive dashboard for real-time exploration
- `visualization/realtime.py`: Matplotlib animations

## Data Output

Simulations save to `data/` by default:
- `*_history.csv`: Tick-by-tick metrics (strategy distribution, trust, coordination rate, norm level)
- `*_results.json`: Configuration + final state summary
- `*_summary.png/.pdf`: Static plots

Experiments save to `data/experiments/`:
- `*_comparison.csv`: Results across all trials
- `*_comparison.json`: Full config + results for each trial

## Key Parameters

When modifying or creating experiments:

**Memory Parameters:**
- `memory_type`: "fixed" | "decay" | "dynamic"
- `memory_size`: Base window size (default: 5)
- `dynamic_base`: Min window for dynamic memory (default: 2)
- `dynamic_max`: Max window for dynamic memory (default: 6, matches Miller's cognitive limit)

**Trust Parameters (CognitiveLockIn/EpsilonGreedy):**
- `initial_trust`: Starting trust (default: 0.5)
- `alpha`: Trust increase rate on correct prediction (default: 0.1)
- `beta`: Trust decay rate on wrong prediction (default: 0.3)
- **Rule of thumb**: β > α creates negativity bias; α = β is symmetric; large values = high sensitivity

**Temperature Parameters (DualFeedback):**
- `initial_tau`: Starting temperature (default: 1.0)
- `tau_min`: Minimum (most committed) (default: 0.1)
- `tau_max`: Maximum (most exploratory) (default: 2.0)
- `cooling_rate`: τ decrease on correct (default: 0.1)
- `heating_penalty`: τ increase on wrong (default: 0.3)

**Convergence:**
- `convergence_threshold`: Fraction for consensus (default: 0.95)
- `convergence_window`: Ticks to maintain threshold (default: 50)

## Testing and Validation

No formal test suite exists yet. For validation:
1. Run quick test: `python main.py --quick --verbose`
2. Verify convergence: Should converge in <200 ticks for 100 agents with dynamic memory
3. Check output files exist in `data/`
4. Compare memory types: `python main.py --experiment` - dynamic should converge fastest

## Common Pitfalls

1. **Agent count must be even**: Environment auto-adjusts, but be aware for analysis
2. **Dual feedback may not converge from 50-50**: Requires high initial prediction accuracy (≥75%) to maintain τ. Use cognitive_lockin instead.
3. **Dynamic memory requires decision mechanism with trust**: Won't work with mechanisms that don't implement `get_trust()` properly
4. **Parallel experiments**: Use `n_workers` parameter carefully - too many workers can cause memory issues
5. **Random seeds**: For reproducibility, always specify `random_seed` in config

## Research Context

Based on literature in:
- **Bicchieri (2006)**: Grammar of Society - normative vs empirical expectations
- **Slovic (1993)**: Trust asymmetry - negativity bias in risk perception
- **Skyrms (2010)**: Signals - pre-play communication in coordination games
- **Miller (1956)**: Cognitive limits - 7±2 items in working memory (we use max=6)

The model extends standard ABM norm formation by making **memory endogenous** - it adapts based on performance rather than being fixed.
