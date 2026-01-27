# ABM Memory & Norm Formation Simulation

Agent-based model exploring how different memory mechanisms affect norm formation in coordination games.

## Overview

This simulation studies the emergence of social norms through the lens of memory and trust. Agents play a pure coordination game where they receive payoff only when choosing the same strategy as their partner. The key innovation is the feedback loop between coordination success, prediction accuracy, and memory mechanisms.

### Feedback Loop

```
Coordination Success → Prediction Correct → τ Decreases → Trust Increases
     ↓                                                          ↓
More Stable Consensus ←←←←←←←← Longer Memory Window ←←←←←←←←←←←←

Coordination Failure → Prediction Wrong → τ Increases → Trust Decreases
     ↓                                                          ↓
System Flexibility ←←←←←←←←←←← Shorter Memory Window ←←←←←←←←←←←←
```

## Installation

```bash
# Clone or download the repository
cd meanField_memory

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Run default simulation (100 agents, fixed memory)
python main.py

# Quick test (10 agents, 100 ticks)
python main.py --quick

# Run with different memory types
python main.py --memory fixed
python main.py --memory decay --decay-rate 0.8
python main.py --memory dynamic --dynamic-max 6

# Custom configuration
python main.py --agents 50 --ticks 500 --seed 123

# Run comparison experiment
python main.py --experiment

# Launch interactive dashboard
python main.py --dashboard
```

## Project Structure

```
meanField_memory/
├── src/
│   ├── agent.py          # Agent class with memory and decision
│   ├── memory/
│   │   ├── base.py       # Base memory class
│   │   ├── fixed.py      # Fixed-length sliding window
│   │   ├── decay.py      # Exponentially decaying weights
│   │   └── dynamic.py    # Trust-linked adaptive window
│   ├── decision.py       # Prediction-error softmax mechanism
│   ├── game.py           # Coordination game logic
│   ├── environment.py    # Simulation environment
│   └── trust.py          # Trust-temperature linkage
├── visualization/
│   ├── realtime.py       # Matplotlib animation
│   ├── static_plots.py   # Publication-quality figures
│   └── dashboard.py      # Streamlit interactive dashboard
├── experiments/
│   └── runner.py         # Batch experiment runner
├── data/                 # Output directory
├── config.py             # Configuration dataclasses
├── main.py               # CLI entry point
└── requirements.txt
```

## Memory Types

### Fixed Memory
- Stores the k most recent interactions
- All interactions weighted equally
- Simple, bounded memory model

### Decay Memory
- Exponentially decaying weights: weight(t) = λ^age
- Recent memories weighted more heavily
- Models natural memory fading

### Dynamic Memory
- Window size adapts based on trust level
- High trust → longer window → more stable
- Low trust → shorter window → more adaptive
- Window range: [2, 6] (cognitive limit)

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| num_agents | 100 | Number of agents (2-200) |
| memory_type | fixed | Memory mechanism |
| memory_size | 5 | Base memory window |
| initial_tau | 1.0 | Starting temperature |
| tau_min | 0.1 | Minimum temperature |
| tau_max | 2.0 | Maximum temperature |
| cooling_rate | 0.1 | τ decrease on success |
| heating_penalty | 0.3 | τ increase on failure |
| max_ticks | 1000 | Simulation length |

## Python API

```python
from src.environment import SimulationEnvironment

# Create and run simulation
env = SimulationEnvironment(
    num_agents=100,
    memory_type="dynamic",
    max_ticks=1000,
)

result = env.run(verbose=True)

# Access results
print(f"Converged: {result.converged}")
print(f"Convergence tick: {result.convergence_tick}")

# Get data as DataFrame
df = env.get_history_dataframe()

# Save results
env.save_results(output_dir="data")
```

## Visualization

```python
from visualization.static_plots import create_summary_figure

# Generate summary figure
fig = create_summary_figure(df)
fig.savefig("summary.png")
```

## Experiments

```python
from experiments.runner import run_memory_comparison, run_agent_count_sweep

# Compare memory types
df = run_memory_comparison(num_agents=100, n_trials=10)

# Sweep agent counts
df = run_agent_count_sweep(
    agent_counts=[10, 50, 100, 200],
    memory_type="dynamic",
)
```

## Key Metrics

- **Convergence Time**: Ticks to reach stable consensus (95%+)
- **Convergence Rate**: Fraction of simulations that converge
- **Consensus Strength**: Final majority fraction
- **Mean Trust**: Average trust level at end
- **Strategy Switches**: Behavioral volatility

## References

Based on research in:
- Agent-based modeling of social norms
- Memory and learning in game theory
- Bounded rationality and cognitive limits
