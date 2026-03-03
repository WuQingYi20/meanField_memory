# Causal Analysis Report — N=20000

Generated: 2026-03-02 17:21:47

---

## S1: Descriptive Overview

**N=20000 | Total runs:** 300 | **Converged:** 157 (52.3%)

**Convergence tick:** mean=127.3, median=118.0, sd=20.0

| θ | Φ | n | conv_rate | mean_diss | mean_enf |
|---|---|---|-----------|-----------|----------|
| 3.0 | 0.0 | 30 | 1.000 | 8164.4 | 0.0 |
| 3.0 | 0.5 | 30 | 1.000 | 8367.3 | 233651.2 |
| 3.0 | 1.0 | 30 | 1.000 | 11542.4 | 261638.2 |
| 3.0 | 2.0 | 30 | 1.000 | 14110.2 | 261848.1 |
| 3.0 | 5.0 | 30 | 1.000 | 22828.8 | 351353.3 |
| 7.0 | 0.0 | 30 | 0.000 | 2151.0 | 0.0 |
| 7.0 | 0.5 | 30 | 0.000 | 2457.2 | 137727.1 |
| 7.0 | 1.0 | 30 | 0.000 | 2720.6 | 139864.0 |
| 7.0 | 2.0 | 30 | 0.000 | 4508.8 | 158173.6 |
| 7.0 | 5.0 | 30 | 0.233 | 11820.1 | 251509.9 |

## S3: Logistic Regression — Cascade Success

| Term | Coef | SE | z | p | OR |
|------|------|----|---|---|----|
| (Intercept) | 49.0625 | 942.8037 | 0.052 | 0.9585 | 2030408642545988665344.000 |
| Phi | 4.9946 | 119.1691 | 0.042 | 0.9666 | 147.621 |
| theta_crystal | -10.7465 | 189.7040 | -0.057 | 0.9548 | 0.000 |

**Pseudo-R²:** 0.9215

## S6: Granger Causality

### Stationarity (ADF)

| Series | ADF_t | Stationary |
|--------|-------|------------|
| anomaly | -5.503 | YES |
| mean_sigma | -1.560 | NO |
| belief_shift | -1.250 | NO |
| mean_C | -0.879 | NO |

### Granger F-tests

| Pair | Lag | F | p | Sig |
|------|-----|---|---|-----|
| anomaly→sigma | 1 | 4.637 | 0.0337 | ** |
| anomaly→sigma | 2 | 3.720 | 0.0278 | ** |
| anomaly→sigma | 3 | 0.854 | 0.4678 |  |
| belief→anomaly | 1 | 0.435 | 0.5111 |  |
| belief→anomaly | 2 | 0.332 | 0.7186 |  |
| belief→anomaly | 3 | 0.363 | 0.7797 |  |
| confidence→sigma | 1 | 2.047 | 0.1557 |  |
| confidence→sigma | 2 | 3.364 | 0.0388 | ** |
| confidence→sigma | 3 | 2.897 | 0.0393 | ** |

## S7: CUSUM Changepoint Detection

**belief:** changepoint at tick 59 (CUSUM=-10.205)

**sigma:** changepoint at tick 25 (CUSUM=0.277)

**confidence:** changepoint at tick 56 (CUSUM=-0.471)

