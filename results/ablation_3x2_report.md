# 3x2 Ablation Experiment: Experiential Memory Level x Normative Memory

**Date**: 2026-02-20  
**Script**: `scripts/run_ablation_3x2.jl`  
**Raw data**: `results/ablation_3x2_summary.csv`  
**Trials**: 100 per cell  

## Experimental Design

3x2 factorial crossed with 3 population sizes (N = 20, 100, 500).

**Factor 1 -- Experiential Memory (3 levels):**
| Level | Description | Parameters |
|-------|-------------|------------|
| None | `b_exp_A` reset to 0.5 after every tick -- no individual learning | -- |
| Fixed | Standard FIFO learning, fixed window | w_base=5, w_max=5 |
| Dynamic | Standard FIFO learning, confidence-driven window | w_base=2, w_max=6 |

**Factor 2 -- Normative Memory:** OFF / ON (`enable_normative`)

**Other parameters:** V=0, Phi=0.0, T=3000, alpha=0.1, beta=0.3, C0=0.5

**Convergence criterion:** Behavioral majority >= 0.95 sustained for 50 consecutive ticks

---
## N = 20

### Convergence Rate and Speed

| Exp Level | Norm OFF conv | Norm OFF mean tick | Norm ON conv | Norm ON mean tick | Speedup |
|-----------|:---:|---:|:---:|---:|---:|
| None (frozen) | 0/100 | -- | 0/100 | -- | -- |
| Fixed (w=5) | 100/100 | 231.6 | 100/100 | 29.9 | 7.7x |
| Dynamic [2,6] | 100/100 | 76.1 | 100/100 | 25.4 | 3.0x |

### Final Majority Fraction (all 100 trials, including non-converged)

| Exp Level | | Mean | Median | Min | Max | Std |
|-----------|------|---:|---:|---:|---:|---:|
| None (frozen) | Norm OFF | 0.598 | 0.6 | 0.5 | 0.85 | 0.077 |
|  | Norm ON | 0.632 | 0.6 | 0.5 | 0.85 | 0.083 |
| Fixed (w=5) | Norm OFF | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 |
|  | Norm ON | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 |
| Dynamic [2,6] | Norm OFF | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 |
|  | Norm ON | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 |

### Final Coordination Rate (all 100 trials)

| Exp Level | | Mean | Median | Min | Max |
|-----------|------|---:|---:|---:|---:|
| None (frozen) | Norm OFF | 0.481 | 0.5 | 0.1 | 0.9 |
|  | Norm ON | 0.521 | 0.5 | 0.1 | 0.9 |
| Fixed (w=5) | Norm OFF | 1.0 | 1.0 | 1.0 | 1.0 |
|  | Norm ON | 1.0 | 1.0 | 1.0 | 1.0 |
| Dynamic [2,6] | Norm OFF | 1.0 | 1.0 | 1.0 | 1.0 |
|  | Norm ON | 1.0 | 1.0 | 1.0 | 1.0 |

---
## N = 100

### Convergence Rate and Speed

| Exp Level | Norm OFF conv | Norm OFF mean tick | Norm ON conv | Norm ON mean tick | Speedup |
|-----------|:---:|---:|:---:|---:|---:|
| None (frozen) | 0/100 | -- | 0/100 | -- | -- |
| Fixed (w=5) | 91/100 | 1125.0 | 100/100 | 42.5 | 26.5x |
| Dynamic [2,6] | 100/100 | 230.7 | 100/100 | 37.1 | 6.2x |

### Final Majority Fraction (all 100 trials, including non-converged)

| Exp Level | | Mean | Median | Min | Max | Std |
|-----------|------|---:|---:|---:|---:|---:|
| None (frozen) | Norm OFF | 0.534 | 0.53 | 0.5 | 0.63 | 0.028 |
|  | Norm ON | 0.615 | 0.62 | 0.51 | 0.76 | 0.052 |
| Fixed (w=5) | Norm OFF | 0.97 | 1.0 | 0.51 | 1.0 | 0.097 |
|  | Norm ON | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 |
| Dynamic [2,6] | Norm OFF | 0.993 | 1.0 | 0.87 | 1.0 | 0.025 |
|  | Norm ON | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 |

### Final Coordination Rate (all 100 trials)

| Exp Level | | Mean | Median | Min | Max |
|-----------|------|---:|---:|---:|---:|
| None (frozen) | Norm OFF | 0.49 | 0.48 | 0.3 | 0.7 |
|  | Norm ON | 0.529 | 0.52 | 0.38 | 0.68 |
| Fixed (w=5) | Norm OFF | 0.961 | 1.0 | 0.42 | 1.0 |
|  | Norm ON | 1.0 | 1.0 | 1.0 | 1.0 |
| Dynamic [2,6] | Norm OFF | 0.987 | 1.0 | 0.76 | 1.0 |
|  | Norm ON | 1.0 | 1.0 | 1.0 | 1.0 |

---
## N = 500

### Convergence Rate and Speed

| Exp Level | Norm OFF conv | Norm OFF mean tick | Norm ON conv | Norm ON mean tick | Speedup |
|-----------|:---:|---:|:---:|---:|---:|
| None (frozen) | 0/100 | -- | 0/100 | -- | -- |
| Fixed (w=5) | 24/100 | 2079.4 | 100/100 | 53.8 | 38.7x |
| Dynamic [2,6] | 100/100 | 467.2 | 100/100 | 43.4 | 10.8x |

### Final Majority Fraction (all 100 trials, including non-converged)

| Exp Level | | Mean | Median | Min | Max | Std |
|-----------|------|---:|---:|---:|---:|---:|
| None (frozen) | Norm OFF | 0.518 | 0.514 | 0.5 | 0.556 | 0.014 |
|  | Norm ON | 0.616 | 0.614 | 0.546 | 0.678 | 0.025 |
| Fixed (w=5) | Norm OFF | 0.787 | 0.797 | 0.5 | 1.0 | 0.159 |
|  | Norm ON | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 |
| Dynamic [2,6] | Norm OFF | 0.957 | 0.958 | 0.914 | 0.988 | 0.017 |
|  | Norm ON | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 |

### Final Coordination Rate (all 100 trials)

| Exp Level | | Mean | Median | Min | Max |
|-----------|------|---:|---:|---:|---:|
| None (frozen) | Norm OFF | 0.501 | 0.504 | 0.42 | 0.58 |
|  | Norm ON | 0.524 | 0.526 | 0.44 | 0.61 |
| Fixed (w=5) | Norm OFF | 0.712 | 0.674 | 0.47 | 1.0 |
|  | Norm ON | 1.0 | 1.0 | 1.0 | 1.0 |
| Dynamic [2,6] | Norm OFF | 0.917 | 0.92 | 0.84 | 0.98 |
|  | Norm ON | 1.0 | 1.0 | 1.0 | 1.0 |

---
## Cross-N Summary

### Convergence rate across N

| Condition | N=20 | N=100 | N=500 |
|-----------|:---:|:---:|:---:|
| None (frozen) | 0/100 | 0/100 | 0/100 |
| None (frozen)+Norm | 0/100 | 0/100 | 0/100 |
| Fixed (w=5) | 100/100 | 91/100 | 24/100 |
| Fixed (w=5)+Norm | 100/100 | 100/100 | 100/100 |
| Dynamic [2,6] | 100/100 | 100/100 | 100/100 |
| Dynamic [2,6]+Norm | 100/100 | 100/100 | 100/100 |

### Mean convergence tick across N (converged trials only)

| Condition | N=20 | N=100 | N=500 |
|-----------|---:|---:|---:|
| None (frozen) | -- | -- | -- |
| None (frozen)+Norm | -- | -- | -- |
| Fixed (w=5) | 231.6 | 1125.0 | 2079.4 |
| Fixed (w=5)+Norm | 29.9 | 42.5 | 53.8 |
| Dynamic [2,6] | 76.1 | 230.7 | 467.2 |
| Dynamic [2,6]+Norm | 25.4 | 37.1 | 43.4 |

### Mean final majority across N (all trials)

| Condition | N=20 | N=100 | N=500 |
|-----------|---:|---:|---:|
| None (frozen) | 0.598 | 0.534 | 0.518 |
| None (frozen)+Norm | 0.632 | 0.615 | 0.616 |
| Fixed (w=5) | 1.0 | 0.97 | 0.787 |
| Fixed (w=5)+Norm | 1.0 | 1.0 | 1.0 |
| Dynamic [2,6] | 1.0 | 0.993 | 0.957 |
| Dynamic [2,6]+Norm | 1.0 | 1.0 | 1.0 |

### Normative speedup ratio across N

| Exp Level | N=20 | N=100 | N=500 |
|-----------|---:|---:|---:|
| Fixed (w=5) | 7.7x | 26.4x | 38.6x |
| Dynamic [2,6] | 3.0x | 6.2x | 10.8x |

---
## Key Findings

### 1. Experiential memory is necessary for convergence
Without experiential learning (None row), convergence rate is 0/100 at all N, regardless of normative memory. With normative memory on, the final majority reaches ~0.62 (vs ~0.53 without) -- a modest improvement from crystallization, but far below the 0.95 convergence threshold. **Norms can amplify existing patterns but cannot create them from scratch.**

### 2. Normative memory dramatically accelerates convergence
When experiential learning is present, adding normative memory reduces convergence time by 3-39x depending on conditions. The speedup is largest when experiential learning is weakest (Fixed at large N): at N=500, Fixed goes from 24/100 converged at 2079 ticks to 100/100 at 54 ticks (38.7x speedup).

### 3. Normative memory compensates for weak experiential learning
Fixed(w=5)+Norm and Dynamic[2,6]+Norm converge at nearly identical speeds (within ~10 ticks across all N). The normative cascade mechanism makes the quality of experiential learning largely irrelevant once it provides a sufficient symmetry-breaking signal.

### 4. Normative memory provides population-size robustness
Without norms, convergence time scales roughly linearly with N (and Fixed even loses convergence reliability at large N: 24% at N=500). With norms, convergence time grows very slowly with N (~25t to ~43t for Dynamic+Norm from N=20 to N=500) and convergence rate stays at 100%.

### 5. Complementary roles
- **Experiential memory** = symmetry breaker: individual learning from partner actions creates a population-level behavioral majority
- **Normative memory** = pattern amplifier: DDM crystallization cascade detects the emerging majority, locks it in as an institutional norm, and drives the population to complete consensus
- Neither system alone achieves reliable, scalable convergence. Together they produce fast, complete, and robust norm emergence.
