# Causal Analysis Report — N=1000

Generated: 2026-03-02 17:21:37

---

## S1: Descriptive Overview

**N=1000 | Total runs:** 300 | **Converged:** 175 (58.3%)

**Convergence tick:** mean=113.3, median=110.0, sd=14.9

| θ | Φ | n | conv_rate | mean_diss | mean_enf |
|---|---|---|-----------|-----------|----------|
| 3.0 | 0.0 | 30 | 1.000 | 312.0 | 0.0 |
| 3.0 | 0.5 | 30 | 1.000 | 226.3 | 7358.6 |
| 3.0 | 1.0 | 30 | 1.000 | 291.0 | 8092.1 |
| 3.0 | 2.0 | 30 | 1.000 | 414.4 | 9887.1 |
| 3.0 | 5.0 | 30 | 1.000 | 658.0 | 12398.9 |
| 7.0 | 0.0 | 30 | 0.033 | 63.3 | 0.0 |
| 7.0 | 0.5 | 30 | 0.167 | 53.7 | 3522.0 |
| 7.0 | 1.0 | 30 | 0.300 | 62.7 | 3708.5 |
| 7.0 | 2.0 | 30 | 0.233 | 73.6 | 3578.7 |
| 7.0 | 5.0 | 30 | 0.100 | 378.8 | 8393.9 |

## S3: Logistic Regression — Cascade Success

| Term | Coef | SE | z | p | OR |
|------|------|----|---|---|----|
| (Intercept) | 28.5030 | 207.9110 | 0.137 | 0.8910 | 2391734100689.106 |
| Phi | -0.0310 | 0.1257 | -0.246 | 0.8054 | 0.969 |
| theta_crystal | -4.2944 | 29.7016 | -0.145 | 0.8850 | 0.014 |

**Pseudo-R²:** 0.6685

## S6: Granger Causality

### Stationarity (ADF)

| Series | ADF_t | Stationary |
|--------|-------|------------|
| anomaly | -4.528 | YES |
| mean_sigma | -3.516 | YES |
| belief_shift | -1.698 | NO |
| mean_C | -3.028 | YES |

### Granger F-tests

| Pair | Lag | F | p | Sig |
|------|-----|---|---|-----|
| anomaly→sigma | 1 | 2.308 | 0.1320 |  |
| anomaly→sigma | 2 | 9.964 | 0.0001 | *** |
| anomaly→sigma | 3 | 6.007 | 0.0009 | *** |
| belief→anomaly | 1 | 0.087 | 0.7691 |  |
| belief→anomaly | 2 | 3.520 | 0.0336 | ** |
| belief→anomaly | 3 | 1.248 | 0.2972 |  |
| confidence→sigma | 1 | 0.361 | 0.5494 |  |
| confidence→sigma | 2 | 2.479 | 0.0894 | * |
| confidence→sigma | 3 | 4.287 | 0.0071 | *** |

## S7: CUSUM Changepoint Detection

**belief:** changepoint at tick 39 (CUSUM=9.292)

**sigma:** changepoint at tick 47 (CUSUM=-8.921)

**confidence:** changepoint at tick 48 (CUSUM=-14.709)

