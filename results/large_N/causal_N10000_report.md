# Causal Analysis Report — N=10000

Generated: 2026-03-02 17:21:45

---

## S1: Descriptive Overview

**N=10000 | Total runs:** 300 | **Converged:** 158 (52.7%)

**Convergence tick:** mean=123.1, median=117.0, sd=18.4

| θ | Φ | n | conv_rate | mean_diss | mean_enf |
|---|---|---|-----------|-----------|----------|
| 3.0 | 0.0 | 30 | 1.000 | 3891.2 | 0.0 |
| 3.0 | 0.5 | 30 | 1.000 | 4062.0 | 114104.2 |
| 3.0 | 1.0 | 30 | 1.000 | 5744.1 | 126836.1 |
| 3.0 | 2.0 | 30 | 1.000 | 5591.5 | 118992.3 |
| 3.0 | 5.0 | 30 | 1.000 | 8998.4 | 148739.8 |
| 7.0 | 0.0 | 30 | 0.000 | 1005.6 | 0.0 |
| 7.0 | 0.5 | 30 | 0.000 | 903.0 | 53748.3 |
| 7.0 | 1.0 | 30 | 0.000 | 969.2 | 53241.3 |
| 7.0 | 2.0 | 30 | 0.033 | 1822.0 | 64461.6 |
| 7.0 | 5.0 | 30 | 0.233 | 5671.3 | 120375.5 |

## S3: Logistic Regression — Cascade Success

| Term | Coef | SE | z | p | OR |
|------|------|----|---|---|----|
| (Intercept) | 31.8310 | 320.5976 | 0.099 | 0.9209 | 66684021024488.438 |
| Phi | 0.9864 | 0.3220 | 3.063 | 0.0022 | 2.681 |
| theta_crystal | -5.4175 | 45.8008 | -0.118 | 0.9058 | 0.004 |

**Pseudo-R²:** 0.8975

## S6: Granger Causality

### Stationarity (ADF)

| Series | ADF_t | Stationary |
|--------|-------|------------|
| anomaly | -4.904 | YES |
| mean_sigma | -3.075 | YES |
| belief_shift | -1.382 | NO |
| mean_C | -1.704 | NO |

### Granger F-tests

| Pair | Lag | F | p | Sig |
|------|-----|---|---|-----|
| anomaly→sigma | 1 | 0.051 | 0.8210 |  |
| anomaly→sigma | 2 | 6.132 | 0.0031 | *** |
| anomaly→sigma | 3 | 7.480 | 0.0002 | *** |
| belief→anomaly | 1 | 0.991 | 0.3220 |  |
| belief→anomaly | 2 | 0.862 | 0.4257 |  |
| belief→anomaly | 3 | 0.795 | 0.4999 |  |
| confidence→sigma | 1 | 5.123 | 0.0258 | ** |
| confidence→sigma | 2 | 1.051 | 0.3536 |  |
| confidence→sigma | 3 | 3.166 | 0.0282 | ** |

## S7: CUSUM Changepoint Detection

**belief:** changepoint at tick 44 (CUSUM=-9.998)

**sigma:** changepoint at tick 52 (CUSUM=-8.370)

**confidence:** changepoint at tick 37 (CUSUM=-0.369)

