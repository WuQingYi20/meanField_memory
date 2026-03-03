# Causal Analysis Report — N=5000

Generated: 2026-03-02 17:21:44

---

## S1: Descriptive Overview

**N=5000 | Total runs:** 300 | **Converged:** 152 (50.7%)

**Convergence tick:** mean=118.8, median=114.0, sd=15.3

| θ | Φ | n | conv_rate | mean_diss | mean_enf |
|---|---|---|-----------|-----------|----------|
| 3.0 | 0.0 | 30 | 1.000 | 1770.9 | 0.0 |
| 3.0 | 0.5 | 30 | 1.000 | 1714.9 | 49741.1 |
| 3.0 | 1.0 | 30 | 1.000 | 2215.3 | 56101.4 |
| 3.0 | 2.0 | 30 | 1.000 | 2893.6 | 60332.9 |
| 3.0 | 5.0 | 30 | 1.000 | 4149.8 | 71088.3 |
| 7.0 | 0.0 | 30 | 0.000 | 544.5 | 0.0 |
| 7.0 | 0.5 | 30 | 0.000 | 348.7 | 21729.9 |
| 7.0 | 1.0 | 30 | 0.033 | 491.8 | 26701.7 |
| 7.0 | 2.0 | 30 | 0.033 | 629.6 | 25394.3 |
| 7.0 | 5.0 | 30 | 0.000 | 2312.7 | 54313.4 |

## S3: Logistic Regression — Cascade Success

| Term | Coef | SE | z | p | OR |
|------|------|----|---|---|----|
| (Intercept) | 34.1004 | 564.7909 | 0.060 | 0.9519 | 645113762202928.500 |
| Phi | -0.0684 | 0.4278 | -0.160 | 0.8730 | 0.934 |
| theta_crystal | -5.4707 | 80.6845 | -0.068 | 0.9459 | 0.004 |

**Pseudo-R²:** 0.9490

## S6: Granger Causality

### Stationarity (ADF)

| Series | ADF_t | Stationary |
|--------|-------|------------|
| anomaly | -4.370 | YES |
| mean_sigma | -3.561 | YES |
| belief_shift | -1.808 | NO |
| mean_C | -2.377 | NO |

### Granger F-tests

| Pair | Lag | F | p | Sig |
|------|-----|---|---|-----|
| anomaly→sigma | 1 | 0.002 | 0.9658 |  |
| anomaly→sigma | 2 | 4.622 | 0.0122 | ** |
| anomaly→sigma | 3 | 6.436 | 0.0005 | *** |
| belief→anomaly | 1 | 0.628 | 0.4300 |  |
| belief→anomaly | 2 | 0.852 | 0.4298 |  |
| belief→anomaly | 3 | 0.663 | 0.5770 |  |
| confidence→sigma | 1 | 3.522 | 0.0636 | * |
| confidence→sigma | 2 | 1.142 | 0.3236 |  |
| confidence→sigma | 3 | 1.679 | 0.1772 |  |

## S7: CUSUM Changepoint Detection

**belief:** changepoint at tick 41 (CUSUM=9.506)

**sigma:** changepoint at tick 49 (CUSUM=-8.472)

**confidence:** changepoint at tick 32 (CUSUM=-0.347)

