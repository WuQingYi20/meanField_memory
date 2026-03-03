# Causal Analysis Report — Norm Emergence Cascades (v2)

Generated: 2026-03-02 13:41:58

---

## S1: Descriptive Overview

| Phi | theta_crystal | n | conv_rate | mean_tick | mean_dom_frac |
|-----|--------------|---|-----------|-----------|---------------|
| 0.0 | 1.5 | 180 | 1.000 | 97.6 | 0.999 |
| 0.0 | 3.0 | 180 | 1.000 | 93.4 | 0.978 |
| 0.0 | 5.0 | 180 | 0.922 | 97.2 | 0.948 |

**Overall convergence rate:** 0.974 (526/540)

## S2: Partial Correlation Graph

> Dropped zero-variance variables: peak_norm_split

> **Note:** Correlation matrix was near-singular (cond=267.9). Applied Tikhonov regularization (λ=0.1) to enable partial correlation estimation.

Significant partial correlations (|r| > 0.15, p < 0.05):

| Var1 | Var2 | partial_r | p_value | sig |
|------|------|-----------|---------|-----|
| Phi | theta_crystal | 0.250 | 0.0001 | *** |
| Phi | time_in_churn | -0.193 | 0.0027 | ** |
| theta_crystal | total_dissolutions | -0.207 | 0.0013 | ** |
| theta_crystal | recryst_flip | -0.261 | 0.0000 | *** |
| theta_crystal | time_in_churn | 0.155 | 0.0163 | * |
| theta_crystal | convergence_tick | 0.481 | 0.0000 | *** |
| total_dissolutions | total_enforcements | 0.297 | 0.0000 | *** |
| total_dissolutions | recryst_flip | 0.366 | 0.0000 | *** |
| total_dissolutions | time_in_churn | 0.448 | 0.0000 | *** |
| total_enforcements | recryst_flip | 0.358 | 0.0000 | *** |
| total_enforcements | time_in_churn | -0.210 | 0.0011 | ** |
| total_enforcements | convergence_tick | 0.248 | 0.0001 | *** |
| recryst_flip | time_in_churn | 0.320 | 0.0000 | *** |

*13 significant partial correlations found (n=243, df=236) (with Tikhonov regularization).*

## S3: Logistic Regression — Cascade Success/Failure

### Model 1: Main Effects

`converged ~ Phi + theta_crystal`

| Term | Coef | SE | z | p | OR |
|------|------|----|---|---|----|
| (Intercept) | 26.8301 | 207.9203 | 0.129 | 0.8973 | 448902267067.893 |
| Phi | 0.0261 | 0.0952 | 0.274 | 0.7838 | 1.026 |
| theta_crystal | -3.7692 | 29.7029 | -0.127 | 0.8990 | 0.023 |

**McFadden pseudo-R²:** 0.3174
**Classification accuracy:** 0.810

### Model 2: With Interaction

`converged ~ Phi * theta_crystal`

| Term | Coef | SE | z | p | OR |
|------|------|----|---|---|----|
| (Intercept) | 26.9066 | 287.7453 | 0.094 | 0.9255 | 484592826179.964 |
| Phi | -0.0196 | 116.9851 | -0.000 | 0.9999 | 0.981 |
| theta_crystal | -3.7802 | 41.1065 | -0.092 | 0.9267 | 0.023 |
| Phi & theta_crystal | 0.0065 | 16.7122 | 0.000 | 0.9997 | 1.007 |

**McFadden pseudo-R²:** 0.3174
**Classification accuracy:** 0.810

**LR test (interaction):** χ²=0.000, p=0.9999

## S4: Linear Regression — Convergence Speed

### 4a: OLS Regression

`log(convergence_tick) ~ Phi + theta_crystal + total_dissolutions + total_enforcements + peak_norm_split + time_in_churn`

| Term | Coef | SE | t | p |
|------|------|----|---|---|
| (Intercept) | 0.0000 | NaN | NaN | NaN |
| Phi | -0.0179 | 0.0046 | -3.844 | 0.0002 |
| theta_crystal | 0.0583 | 0.0050 | 11.734 | 0.0000 |
| total_dissolutions | 0.0087 | 0.0021 | 4.036 | 0.0001 |
| total_enforcements | 0.0001 | 0.0001 | 2.691 | 0.0076 |
| peak_norm_split | 4.2094 | 0.0272 | 154.683 | 0.0000 |
| time_in_churn | -0.0108 | 0.0054 | -1.982 | 0.0487 |

**R²:** 0.5490 | **Adj R²:** 0.5395

### Standardized Coefficients

| Term | Std_Coef | Relative_Importance |
|------|----------|--------------------|
| Phi | -0.1922 | 0.192 |
| theta_crystal | 0.6926 | 0.693 |
| total_dissolutions | 0.8612 | 0.861 |
| total_enforcements | 0.3248 | 0.325 |
| peak_norm_split | 0.0000 | 0.000 |
| time_in_churn | -0.2459 | 0.246 |

### Variance Inflation Factors (VIF)

| Term | VIF | Concern |
|------|-----|--------|
| Phi | 1.31 | ok |
| theta_crystal | 1.83 | ok |
| total_dissolutions | 23.93 | HIGH |
| total_enforcements | 7.65 | moderate |
| peak_norm_split | 1.00 | ok |
| time_in_churn | 8.09 | moderate |

> **Multicollinearity warning:** Variables total_dissolutions have VIF > 10. OLS coefficient estimates for these variables are unreliable. See Ridge regression (S4b) and PCA regression (S4c) below for robust alternatives.

### 4b: Ridge Regression (L2 regularization)

Addresses multicollinearity by penalizing large coefficients.

> Dropped zero-variance variables for Ridge: peak_norm_split

**Optimal λ (GCV):** 1.000
**Ridge R²:** 0.5487

| Term | Ridge_Coef (std) | OLS_Coef (std) | Shrinkage |
|------|-----------------|----------------|----------|
| Phi | -0.0302 | -0.0315 | 4.3% |
| theta_crystal | 0.1110 | 0.1136 | 2.3% |
| total_dissolutions | 0.1280 | 0.1412 | 9.4% |
| total_enforcements | 0.0583 | 0.0533 | -9.5% |
| time_in_churn | -0.0330 | -0.0403 | 18.2% |

**Robustness verdict:** 5/5 coefficients retain the same sign under Ridge. The driver ranking from OLS is **robust** despite multicollinearity.

### 4c: PCA Regression (collinear variables consolidated)

**Principal Components:**

| PC | Eigenvalue | Var_Explained | Cumulative |
|----|------------|---------------|------------|
| PC1 | 3.0396 | 0.608 | 0.608 |
| PC2 | 1.0257 | 0.205 | 0.813 |
| PC3 | 0.6327 | 0.127 | 0.940 |
| PC4 | 0.2745 | 0.055 | 0.994 |
| PC5 | 0.0276 | 0.006 | 1.000 |

Using first 4 PCs (99.4% variance).
**PCA regression R²:** 0.5333 | **Adj R²:** 0.5255

**PC loadings (top variables per component):**

| Variable | PC1 | PC2 | PC3 | PC4 |
|----------|---|---|---|---|
| Phi | 0.183 | 0.902 | 0.255 | 0.294 |
| theta_crystal | -0.377 | 0.397 | -0.778 | -0.292 |
| total_dissolutions | 0.563 | -0.011 | -0.161 | -0.036 |
| total_enforcements | 0.522 | 0.107 | 0.033 | -0.750 |
| time_in_churn | 0.484 | -0.134 | -0.550 | 0.514 |

## S5: Mediation Analysis — Baron-Kenny + Bootstrap

**Path:** Phi → total_enforcements → total_dissolutions → convergence_tick

> Bootstrap CI (percentile method, B=2000) supplements the Sobel test as it does not assume normality of the indirect effect distribution.

### Segment 1: Phi → total_enforcements → log_conv_tick

| Path | Coef | p |
|------|------|---|
| c (total) | 0.0124 | - |
| a (Phi→enforcements) | 68.6411 | - |
| b (enforcements→tick) | 0.0002 | - |
| c' (direct) | -0.0041 | - |

**Indirect effect (a×b):** 0.0164
**Sobel test:** z=4.701, p=0.0000
**Bootstrap 95% CI:** [0.0098, 0.0244] (B=2000)
**Bootstrap mean:** 0.0165, SE=0.0038
**Bootstrap significant (CI excludes 0):** YES
**Proportion mediated:** 1.330

### Segment 2: Phi → total_dissolutions → log_conv_tick

| Path | Coef | p |
|------|------|---|
| c (total) | 0.0124 | - |
| a (Phi→dissolutions) | 2.5144 | - |
| b (dissolutions→tick) | 0.0049 | - |
| c' (direct) | -0.0001 | - |

**Indirect effect (a×b):** 0.0124
**Sobel test:** z=3.896, p=0.0001
**Bootstrap 95% CI:** [0.0054, 0.0197] (B=2000)
**Bootstrap mean:** 0.0124, SE=0.0036
**Bootstrap significant (CI excludes 0):** YES
**Proportion mediated:** 1.007

## S6: Granger Causality (VAR F-test)

> **Caveat:** Granger causality tests *temporal predictive precedence*, not physical causation. A significant result means past values of X improve prediction of Y beyond Y's own past, which is a necessary but not sufficient condition for true causality.

### Stationarity Tests (ADF)

> Granger causality requires stationary series. Non-stationary data can produce spurious regressions.

| Series | ADF_t | Lag | 5%_CV | Stationary |
|--------|-------|-----|-------|------------|
| anomaly | -3.039 | 2 | -2.86 | YES |
| mean_sigma | -3.648 | 3 | -2.86 | YES |
| belief_shift | -4.641 | 1 | -2.86 | YES |
| mean_C | -2.533 | 2 | -2.86 | NO → use Δ |
| mean_beff_A | -2.680 | 3 | -2.86 | NO → use Δ |
| norm_strength | -2.533 | 2 | -2.86 | NO → use Δ |

**Non-stationary series:** mean_C, mean_beff_A, norm_strength. These are first-differenced before Granger testing below.

### Optimal Lag Selection (AIC/BIC)

| Pair | AIC_lag | BIC_lag |
|------|---------|--------|
| anomaly → dissolution_proxy(sigma) | 4 | 4 |
| belief_shift → anomaly | 3 | 3 |
| confidence → norm_strength | 2 | 2 |
| sigma → belief | 4 | 1 |

### Granger F-tests

| Cause → Effect | Lag | F-stat | p-value | Sig | Optimal |
|----------------|-----|--------|---------|-----|--------|
| anomaly → dissolution_proxy(sigma) | 1 | 138.067 | 0.0000 | *** |  |
| anomaly → dissolution_proxy(sigma) | 2 | 43.171 | 0.0000 | *** |  |
| anomaly → dissolution_proxy(sigma) | 3 | 18.281 | 0.0000 | *** |  |
| belief_shift → anomaly | 1 | 4.747 | 0.0301 | ** |  |
| belief_shift → anomaly | 2 | 1.846 | 0.1596 |  |  |
| belief_shift → anomaly | 3 | 0.892 | 0.4454 |  | ← |
| confidence → norm_strength | 1 | 0.000 | 1.0000 |  |  |
| confidence → norm_strength | 2 | 0.000 | 1.0000 |  | ← |
| confidence → norm_strength | 3 | 0.000 | 1.0000 |  |  |
| sigma → belief | 1 | 10.858 | 0.0011 | *** | ← |
| sigma → belief | 2 | 8.096 | 0.0004 | *** |  |
| sigma → belief | 3 | 3.432 | 0.0175 | ** |  |

## S7: Changepoint Detection (CUSUM)

| Series | Changepoint_tick | CUSUM_max | Direction |
|--------|-----------------|-----------|----------|
| Crystallized agents | 86 | 133.949 | decrease |
| Mean belief (beff_A) | 67 | 119.873 | decrease |
| Mean sigma | 84 | 117.086 | decrease |

### Comparison with Known Phase Boundaries

| Phase | Mean_tick | Median_tick | SD |
|-------|----------|-------------|----|
| first_tick_behavioral | 39.1 | 35.0 | 16.2 |
| first_tick_belief | 39.6 | 36.0 | 16.2 |
| first_tick_crystal | 45.4 | 44.0 | 15.2 |
| first_tick_all_met | 47.1 | 45.0 | 16.5 |

### Sensitivity Analysis: Changepoint Drift by theta_crystal

> How do changepoint timings shift when the crystallization threshold changes?

| theta_crystal | conv_rate | mean_conv_tick | mean_first_belief | mean_first_crystal |
|--------------|-----------|---------------|-------------------|-------------------|
| 1.5 | 1.000 | 97.6 | 41.3 | 43.2 |
| 3.0 | 1.000 | 93.4 | 35.5 | 40.6 |
| 5.0 | 0.922 | 105.4 | 42.2 | 52.9 |

### Early Warning Signals (EWS)

> Can the belief shift at the detected changepoint serve as a leading indicator for the subsequent crystallization cascade?

- **Belief changepoint:** tick 67
- **Crystallization changepoint:** tick 86
- **Lead time:** 19 ticks

The belief shift precedes the crystallization cascade by 19 ticks, suggesting it can function as an **early warning signal**.

**Rolling variance trend (pre-changepoint):** r=0.877 (INCREASING — consistent with critical slowing down)

## S8: Counterfactual Intervention Analysis

> Using the estimated causal model (structural equations from S4/S5), we simulate *do-interventions*: what would happen if we externally set a variable to a specific value, breaking its natural correlations?

### Intervention 1: do(Phi := Phi - 5)

"What if environmental pressure were reduced by 5 units?"

| Variable | Observed_Mean | Counterfactual | Δ |
|----------|--------------|----------------|---|
| Phi | 1.7 | -3.3 | -5.0 |
| total_enforcements | 322.8 | -25.0 | -347.7 |
| total_dissolutions | 12.3 | -0.5 | -12.8 |
| log(conv_tick) | 4.576 | 4.514 | -0.061 |
| conv_tick | 97.1 | 91.3 | -5.9% |

### Intervention 2: do(theta_crystal := theta_crystal + 1)

"What if the crystallization threshold were raised by 1?"

| Variable | Observed_Mean | Counterfactual | Δ |
|----------|--------------|----------------|---|
| theta_crystal | 4.5 | 5.5 | 1.0 |
| total_enforcements | 322.8 | 227.1 | -95.6 |
| total_dissolutions | 12.3 | 7.5 | -4.8 |
| log(conv_tick) | 4.576 | 4.587 | 0.011 |
| conv_tick | 97.1 | 98.2 | +1.1% |

### Intervention 3: do(total_enforcements := 0)

"What if norm enforcement were completely disabled?"

| Variable | Observed_Mean | Counterfactual | Δ |
|----------|--------------|----------------|---|
| total_enforcements | 322.8 | 0.0 | -322.8 |
| total_dissolutions | 12.3 | 0.5 | -11.8 |
| log(conv_tick) | 4.576 | 4.450 | -0.126 |
| conv_tick | 97.1 | 85.6 | -11.8% |

> **Validity note:** These counterfactuals assume the linear structural equations remain valid under intervention (no model misspecification, no unobserved confounders). The estimates should be treated as first-order approximations. For large interventions, nonlinear effects may dominate.

## S9: Visualizations

### Causal DAG

![Causal DAG](causal_dag.png)

Nodes represent key variables; directed edges represent estimated causal paths from regression analysis (S4/S5). Edge labels show standardized effect sizes.

### Changepoint Timeline

![Changepoint Timeline](causal_changepoint_timeline.png)

Vertical dashed lines mark CUSUM-detected changepoints. Shaded regions indicate the early warning window between belief shift and crystallization cascade.

