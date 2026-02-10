# Project Summary: Cognitive Lock-in ABM

## Quick Overview

**Project Goal**: Understand how cognitive mechanisms (memory + trust) drive social norm emergence

**Core Innovation**: Cognitive Lock-in - a feedback loop where trust regulates memory, creating norm persistence

**Target**: EUMAS 2025 / AAMAS 2026

---

## What's Built

### Core Components
| Component | Status | File |
|-----------|--------|------|
| Fixed Memory | Done | `src/memory/fixed.py` |
| Decay Memory | Done | `src/memory/decay.py` |
| Dynamic Memory | Done | `src/memory/dynamic.py` |
| Cognitive Lock-in Decision | Done | `src/decision/cognitive_lockin.py` |
| Dual Feedback Decision | Done | `src/decision/dual_feedback.py` |
| Normative Signaling | Done | `src/communication/mechanisms.py` |
| Pre-play Signaling | Done | `src/communication/mechanisms.py` |
| Threshold Contagion | Done | `src/communication/mechanisms.py` |
| Base Environment | Done | `src/environment.py` |
| Extended Environment | Done | `src/environment_extended.py` |
| Experiment Runner | Done | `experiments/runner.py` |

### Key Findings
1. **Dynamic memory > Fixed memory**: 17% faster convergence, 4% higher consensus
2. **Trust asymmetry matters**: beta/alpha ratio of 3:1 is optimal
3. **Scale-efficient**: Convergence time ~ O(N^0.6)
4. **Path-dependent**: Either strategy can win (no bias after fix)

---

## Key Equations

**Trust Update**:
```
Success: T_new = T + alpha * (1 - T)
Failure: T_new = T * (1 - beta)
```

**Steady State**:
```
T* = (p * alpha) / (p * alpha + (1-p) * beta)
```

**Dynamic Window**:
```
window = base + round(trust * (max - base))
```

---

## Next Steps

### Immediate (EUMAS)
1. Communication mechanism comparison experiments
2. Parameter sensitivity analysis
3. Write paper draft

### Medium-term (AAMAS)
1. Network structure extension
2. Heterogeneous agents
3. Multi-strategy games

---

## File Structure

```
meanField_memory/
├── src/
│   ├── agent.py
│   ├── memory/
│   │   ├── fixed.py
│   │   ├── decay.py
│   │   └── dynamic.py
│   ├── decision/
│   │   ├── cognitive_lockin.py
│   │   └── dual_feedback.py
│   ├── communication/
│   │   ├── mechanisms.py
│   │   └── observation.py
│   ├── environment.py
│   └── environment_extended.py
├── experiments/
│   └── runner.py
├── visualization/
│   ├── realtime.py
│   └── static_plots.py
└── docs/
    ├── EUMAS_AAMAS_Paper_Draft.md  <-- Main paper
    ├── Research_Roadmap.md          <-- Future directions
    └── Project_Summary.md           <-- This file
```

---

## Running Experiments

```python
from src.environment import SimulationEnvironment
from src.decision import DecisionMode

# Basic experiment
env = SimulationEnvironment(
    num_agents=100,
    memory_type="dynamic",
    decision_mode=DecisionMode.COGNITIVE_LOCKIN,
    initial_trust=0.5,
    alpha=0.1,
    beta=0.3,
)

result = env.run(max_ticks=500)
print(f"Converged: {result.converged}")
print(f"Time: {result.convergence_time}")
print(f"Majority: {result.final_majority}")
```

```python
# With communication mechanisms
from src.environment_extended import ExtendedEnvironment

env = ExtendedEnvironment(
    num_agents=100,
    enable_normative=True,
    enable_preplay=True,
    enable_threshold=False,
)

result = env.run(max_ticks=500)
```

---

## Contact

For questions about this project, refer to the detailed documentation in `docs/`.
