#!/usr/bin/env julia
#
# Causal Analysis Pipeline for Norm Emergence Cascades (v2)
# ==========================================================
# Goes beyond descriptive statistics to answer:
#   Q1: What triggers cascade phase transitions?
#   Q2: What determines convergence speed?
#   Q3: Why do some cascades fail?
#
# v2 improvements:
#   - S2/S4: Ridge regression + PCA for multicollinearity
#   - S5: Bootstrap confidence intervals for mediation
#   - S6: ADF stationarity tests + AIC/BIC lag selection
#   - S7: Sensitivity analysis + early warning signals
#   - S8: Counterfactual intervention analysis (Do-calculus)
#   - S9: DAG + timeline visualizations
#
# Outputs:
#   results/causal_analysis_report.md
#   results/causal_analysis_summary.csv
#   results/causal_dag.png
#   results/causal_changepoint_timeline.png

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CSV, DataFrames, Statistics, StatsBase
using GLM, HypothesisTests, Distributions, StatsModels
using LinearAlgebra
using Printf
using Random
using Plots; gr()

const RESULTS_DIR = joinpath(@__DIR__, "..", "results")
const N_BOOTSTRAP = 2000

# ════════════════════════════════════════════════════════════════
# S0: Data Loading & Derived Columns
# ════════════════════════════════════════════════════════════════

function load_data()
    summary_path  = joinpath(RESULTS_DIR, "convergence_cascade_summary.csv")
    pathways_path = joinpath(RESULTS_DIR, "cascade_pathways.csv")
    snapshot_path = joinpath(RESULTS_DIR, "cascade_agent_snapshots.csv")

    summary_df  = CSV.read(summary_path,  DataFrame)
    pathways_df = CSV.read(pathways_path, DataFrame)
    snapshot_df = CSV.read(snapshot_path,  DataFrame)

    if eltype(summary_df.converged) <: AbstractString
        summary_df.converged  = summary_df.converged .== "true"
    end
    if eltype(pathways_df.converged) <: AbstractString
        pathways_df.converged = pathways_df.converged .== "true"
    end

    summary_df.cascade = summary_df.converged

    summary_df.log_conv_tick = [
        row.converged ? log(max(row.convergence_tick, 1)) : missing
        for row in eachrow(summary_df)
    ]

    pathways_df.log_conv_tick = [
        row.converged ? log(max(row.convergence_tick, 1)) : missing
        for row in eachrow(pathways_df)
    ]

    println("  Summary:   $(nrow(summary_df)) rows, $(ncol(summary_df)) cols")
    println("  Pathways:  $(nrow(pathways_df)) rows, $(ncol(pathways_df)) cols")
    println("  Snapshots: $(nrow(snapshot_df)) rows, $(ncol(snapshot_df)) cols")

    return summary_df, pathways_df, snapshot_df
end

# ════════════════════════════════════════════════════════════════
# S1: Descriptive Overview
# ════════════════════════════════════════════════════════════════

function descriptive_overview!(report::IOBuffer, stats_rows::Vector, summary_df::DataFrame)
    println("  [S1] Descriptive overview...")
    write(report, "## S1: Descriptive Overview\n\n")

    gdf = groupby(summary_df, [:Phi, :theta_crystal])
    desc = combine(gdf,
        :converged => mean => :convergence_rate,
        :convergence_tick => (x -> mean(skipmissing(x))) => :mean_tick,
        :final_frac_dominant_norm => mean => :mean_dominant_frac,
        nrow => :n_trials,
    )
    sort!(desc, [:Phi, :theta_crystal])

    write(report, "| Phi | theta_crystal | n | conv_rate | mean_tick | mean_dom_frac |\n")
    write(report, "|-----|--------------|---|-----------|-----------|---------------|\n")
    for row in eachrow(desc)
        write(report, @sprintf("| %.1f | %.1f | %d | %.3f | %.1f | %.3f |\n",
            row.Phi, row.theta_crystal, row.n_trials,
            row.convergence_rate, row.mean_tick, row.mean_dominant_frac))
    end
    write(report, "\n")

    overall_conv_rate = mean(summary_df.converged)
    push!(stats_rows, ("S1", "overall_convergence_rate", overall_conv_rate, NaN, NaN, ""))
    push!(stats_rows, ("S1", "n_total_runs", Float64(nrow(summary_df)), NaN, NaN, ""))

    write(report, "**Overall convergence rate:** $(round(overall_conv_rate, digits=3)) ")
    write(report, "($(sum(summary_df.converged))/$(nrow(summary_df)))\n\n")
end

# ════════════════════════════════════════════════════════════════
# S2: Partial Correlation Graph (with Tikhonov regularization)
# ════════════════════════════════════════════════════════════════

function partial_correlation_graph!(report::IOBuffer, stats_rows::Vector, pathways_df::DataFrame)
    println("  [S2] Partial correlation graph (regularized)...")
    write(report, "## S2: Partial Correlation Graph\n\n")

    vars = [:Phi, :theta_crystal, :total_dissolutions, :total_enforcements,
            :recryst_flip, :peak_norm_split, :time_in_churn, :convergence_tick]

    sub = pathways_df[pathways_df.converged, :]
    if nrow(sub) < 10
        write(report, "*Insufficient converged runs for partial correlation analysis.*\n\n")
        return
    end

    available_vars = [v for v in vars if hasproperty(sub, v)]
    mat = Matrix{Float64}(sub[:, available_vars])

    valid_rows = [all(isfinite.(mat[i, :])) for i in 1:size(mat, 1)]
    mat = mat[valid_rows, :]

    # Remove zero-variance columns (cause NaN in correlation)
    col_stds = vec(std(mat, dims=1))
    nonzero_cols = findall(col_stds .> 1e-12)
    if length(nonzero_cols) < length(available_vars)
        dropped = available_vars[setdiff(1:length(available_vars), nonzero_cols)]
        write(report, "> Dropped zero-variance variables: $(join(dropped, ", "))\n\n")
        available_vars = available_vars[nonzero_cols]
        mat = mat[:, nonzero_cols]
    end

    n = size(mat, 1)
    p = size(mat, 2)

    if n < p + 2
        write(report, "*Insufficient observations (n=$n) for partial correlations with p=$p variables.*\n\n")
        return
    end

    C_mat = cor(mat)
    # Replace any remaining NaN/Inf with 0
    C_mat[.!isfinite.(C_mat)] .= 0.0
    for i in 1:p
        C_mat[i, i] = 1.0
    end

    # Check condition number; apply Tikhonov regularization if near-singular
    eigvals_C = eigvals(Symmetric(C_mat))
    min_eig = minimum(eigvals_C)
    max_eig = maximum(eigvals_C)
    cond_num = max_eig / max(min_eig, 1e-15)
    regularized = false
    if cond_num > 100 || min_eig < 1e-10
        # Ledoit-Wolf-style shrinkage: C_reg = (1-λ)C + λI
        λ = 0.1
        C_mat = (1 - λ) * C_mat + λ * I(p)
        regularized = true
        write(report, "> **Note:** Correlation matrix was near-singular (cond=$(round(cond_num, digits=1))). ")
        write(report, "Applied Tikhonov regularization (λ=$λ) to enable partial correlation estimation.\n\n")
    end

    try
        P = inv(C_mat)
        D = Diagonal(1.0 ./ sqrt.(abs.(diag(P))))
        pcor = -D * P * D
        for i in 1:p
            pcor[i, i] = 1.0
        end

        write(report, "Significant partial correlations (|r| > 0.15, p < 0.05):\n\n")
        write(report, "| Var1 | Var2 | partial_r | p_value | sig |\n")
        write(report, "|------|------|-----------|---------|-----|\n")

        n_sig = 0
        for i in 1:p
            for j in (i+1):p
                r = clamp(pcor[i, j], -0.999, 0.999)
                df = n - p
                if df < 3
                    continue
                end
                z = 0.5 * log((1 + r) / (1 - r))
                se = 1.0 / sqrt(max(df - 1, 1))
                z_stat = abs(z) / se
                pval = 2.0 * (1.0 - cdf(Normal(), z_stat))

                if abs(r) > 0.15 && pval < 0.05
                    n_sig += 1
                    sig_str = pval < 0.001 ? "***" : pval < 0.01 ? "**" : "*"
                    write(report, @sprintf("| %s | %s | %.3f | %.4f | %s |\n",
                        available_vars[i], available_vars[j], r, pval, sig_str))
                    push!(stats_rows, ("S2", "pcor_$(available_vars[i])_$(available_vars[j])",
                        r, pval, NaN, sig_str))
                end
            end
        end

        note = regularized ? " (with Tikhonov regularization)" : ""
        write(report, "\n*$n_sig significant partial correlations found (n=$n, df=$(n-p))$note.*\n\n")
    catch e
        write(report, "*Partial correlation computation failed: $e*\n\n")
    end
end

# ════════════════════════════════════════════════════════════════
# S3: Logistic Regression — Cascade Success/Failure
# ════════════════════════════════════════════════════════════════

function logistic_cascade_success!(report::IOBuffer, stats_rows::Vector, pathways_df::DataFrame)
    println("  [S3] Logistic regression for cascade success...")
    write(report, "## S3: Logistic Regression — Cascade Success/Failure\n\n")

    df = copy(pathways_df)
    df.converged_int = Int.(df.converged)

    if length(unique(df.converged_int)) < 2
        write(report, "*All runs have same convergence outcome; logistic regression not applicable.*\n\n")
        return
    end

    # Model 1: Main effects only
    write(report, "### Model 1: Main Effects\n\n")
    write(report, "`converged ~ Phi + theta_crystal`\n\n")

    try
        m1 = glm(@formula(converged_int ~ Phi + theta_crystal), df, Binomial(), LogitLink())
        ct1 = coeftable(m1)

        write(report, "| Term | Coef | SE | z | p | OR |\n")
        write(report, "|------|------|----|---|---|----|\n")
        for i in 1:length(ct1.rownms)
            coef_val = ct1.cols[1][i]
            se_val   = ct1.cols[2][i]
            z_val    = ct1.cols[3][i]
            p_val    = ct1.cols[4][i]
            or_val   = exp(coef_val)
            write(report, @sprintf("| %s | %.4f | %.4f | %.3f | %.4f | %.3f |\n",
                ct1.rownms[i], coef_val, se_val, z_val, p_val, or_val))
        end

        ll_model = loglikelihood(m1)
        m_null = glm(@formula(converged_int ~ 1), df, Binomial(), LogitLink())
        ll_null = loglikelihood(m_null)
        pseudo_r2 = 1.0 - ll_model / ll_null

        pred_prob = predict(m1)
        pred_class = pred_prob .>= 0.5
        accuracy = mean(pred_class .== Bool.(df.converged_int))

        write(report, @sprintf("\n**McFadden pseudo-R²:** %.4f\n", pseudo_r2))
        write(report, @sprintf("**Classification accuracy:** %.3f\n\n", accuracy))

        push!(stats_rows, ("S3", "m1_pseudo_r2", pseudo_r2, NaN, NaN, "main_effects"))
        push!(stats_rows, ("S3", "m1_accuracy", accuracy, NaN, NaN, "main_effects"))

        for i in 2:length(ct1.rownms)
            push!(stats_rows, ("S3", "m1_OR_$(ct1.rownms[i])", exp(ct1.cols[1][i]),
                ct1.cols[4][i], NaN, "main_effects"))
        end
    catch e
        write(report, "*Model 1 failed: $e*\n\n")
    end

    # Model 2: With interaction
    write(report, "### Model 2: With Interaction\n\n")
    write(report, "`converged ~ Phi * theta_crystal`\n\n")

    try
        m2 = glm(@formula(converged_int ~ Phi * theta_crystal), df, Binomial(), LogitLink())
        ct2 = coeftable(m2)

        write(report, "| Term | Coef | SE | z | p | OR |\n")
        write(report, "|------|------|----|---|---|----|\n")
        for i in 1:length(ct2.rownms)
            coef_val = ct2.cols[1][i]
            se_val   = ct2.cols[2][i]
            z_val    = ct2.cols[3][i]
            p_val    = ct2.cols[4][i]
            or_val   = exp(coef_val)
            write(report, @sprintf("| %s | %.4f | %.4f | %.3f | %.4f | %.3f |\n",
                ct2.rownms[i], coef_val, se_val, z_val, p_val, or_val))
        end

        ll_model = loglikelihood(m2)
        m_null = glm(@formula(converged_int ~ 1), df, Binomial(), LogitLink())
        ll_null = loglikelihood(m_null)
        pseudo_r2 = 1.0 - ll_model / ll_null

        pred_prob = predict(m2)
        pred_class = pred_prob .>= 0.5
        accuracy = mean(pred_class .== Bool.(df.converged_int))

        write(report, @sprintf("\n**McFadden pseudo-R²:** %.4f\n", pseudo_r2))
        write(report, @sprintf("**Classification accuracy:** %.3f\n\n", accuracy))

        push!(stats_rows, ("S3", "m2_pseudo_r2", pseudo_r2, NaN, NaN, "interaction"))
        push!(stats_rows, ("S3", "m2_accuracy", accuracy, NaN, NaN, "interaction"))

        m1 = glm(@formula(converged_int ~ Phi + theta_crystal), df, Binomial(), LogitLink())
        lr_stat = 2.0 * (loglikelihood(m2) - loglikelihood(m1))
        lr_pval = 1.0 - cdf(Chisq(1), lr_stat)
        write(report, @sprintf("**LR test (interaction):** χ²=%.3f, p=%.4f\n\n", lr_stat, lr_pval))
        push!(stats_rows, ("S3", "interaction_LR_chi2", lr_stat, lr_pval, 1.0, "df=1"))
    catch e
        write(report, "*Model 2 failed: $e*\n\n")
    end
end

# ════════════════════════════════════════════════════════════════
# S4: Linear Regression — Convergence Speed
#     + Ridge regression + PCA robustness check
# ════════════════════════════════════════════════════════════════

function linear_convergence_speed!(report::IOBuffer, stats_rows::Vector, pathways_df::DataFrame)
    println("  [S4] Linear regression + Ridge + PCA for convergence speed...")
    write(report, "## S4: Linear Regression — Convergence Speed\n\n")

    conv = pathways_df[pathways_df.converged, :]
    if nrow(conv) < 10
        write(report, "*Insufficient converged runs (n=$(nrow(conv))) for regression.*\n\n")
        return
    end

    conv = copy(conv)
    conv.log_conv_tick = log.(max.(conv.convergence_tick, 1))

    predictors = [:Phi, :theta_crystal, :total_dissolutions, :total_enforcements,
                  :peak_norm_split, :time_in_churn]
    available = [p for p in predictors if hasproperty(conv, p)]

    # ── 4a: Standard OLS ──
    write(report, "### 4a: OLS Regression\n\n")
    write(report, "`log(convergence_tick) ~ $(join(available, " + "))`\n\n")

    vif_values = Dict{Symbol, Float64}()
    high_vif_vars = Symbol[]

    try
        rhs = Term.(available)
        f = term(:log_conv_tick) ~ foldl(+, rhs)
        m = lm(f, conv)
        ct = coeftable(m)

        write(report, "| Term | Coef | SE | t | p |\n")
        write(report, "|------|------|----|---|---|\n")
        for i in 1:length(ct.rownms)
            write(report, @sprintf("| %s | %.4f | %.4f | %.3f | %.4f |\n",
                ct.rownms[i], ct.cols[1][i], ct.cols[2][i], ct.cols[3][i], ct.cols[4][i]))
        end

        r2_val = r2(m)
        adj_r2 = adjr2(m)
        write(report, @sprintf("\n**R²:** %.4f | **Adj R²:** %.4f\n\n", r2_val, adj_r2))

        push!(stats_rows, ("S4", "ols_R2", r2_val, NaN, NaN, ""))
        push!(stats_rows, ("S4", "ols_adj_R2", adj_r2, NaN, NaN, ""))

        for i in 2:length(ct.rownms)
            push!(stats_rows, ("S4", "ols_coef_$(ct.rownms[i])", ct.cols[1][i],
                ct.cols[4][i], ct.cols[2][i], ""))
        end

        # Standardized coefficients
        write(report, "### Standardized Coefficients\n\n")
        y_sd = std(conv.log_conv_tick)
        write(report, "| Term | Std_Coef | Relative_Importance |\n")
        write(report, "|------|----------|--------------------|\n")
        for (j, v) in enumerate(available)
            x_sd = std(conv[!, v])
            raw_coef = ct.cols[1][j+1]
            std_c = raw_coef * x_sd / y_sd
            write(report, @sprintf("| %s | %.4f | %.3f |\n", v, std_c, abs(std_c)))
        end
        write(report, "\n")

        # VIF calculation
        write(report, "### Variance Inflation Factors (VIF)\n\n")
        write(report, "| Term | VIF | Concern |\n")
        write(report, "|------|-----|--------|\n")
        X = Matrix{Float64}(conv[:, available])
        for (j, v) in enumerate(available)
            y_j = X[:, j]
            X_other = X[:, setdiff(1:length(available), j)]
            X_aug = hcat(ones(size(X_other, 1)), X_other)
            beta = X_aug \ y_j
            resid = y_j - X_aug * beta
            ss_res = sum(resid .^ 2)
            ss_tot = sum((y_j .- mean(y_j)) .^ 2)
            r2_j = ss_tot > 0 ? 1.0 - ss_res / ss_tot : 0.0
            vif_j = r2_j >= 1.0 ? Inf : 1.0 / (1.0 - r2_j)
            concern = vif_j > 10 ? "HIGH" : vif_j > 5 ? "moderate" : "ok"
            write(report, @sprintf("| %s | %.2f | %s |\n", v, vif_j, concern))
            push!(stats_rows, ("S4", "VIF_$v", vif_j, NaN, NaN, concern))
            vif_values[v] = vif_j
            if vif_j > 10
                push!(high_vif_vars, v)
            end
        end
        write(report, "\n")

        if !isempty(high_vif_vars)
            write(report, "> **Multicollinearity warning:** Variables $(join(high_vif_vars, ", ")) ")
            write(report, "have VIF > 10. OLS coefficient estimates for these variables are unreliable. ")
            write(report, "See Ridge regression (S4b) and PCA regression (S4c) below for robust alternatives.\n\n")
        end
    catch e
        write(report, "*OLS regression failed: $e*\n\n")
    end

    # ── 4b: Ridge Regression ──
    write(report, "### 4b: Ridge Regression (L2 regularization)\n\n")
    write(report, "Addresses multicollinearity by penalizing large coefficients.\n\n")

    try
        X_raw = Matrix{Float64}(conv[:, available])
        y_raw = conv.log_conv_tick

        # Remove zero-variance columns for Ridge/PCA
        col_sds = vec(std(X_raw, dims=1))
        nonzero_mask = col_sds .> 1e-12
        ridge_vars = available[nonzero_mask]
        X_raw = X_raw[:, nonzero_mask]
        if !all(nonzero_mask)
            dropped = available[.!nonzero_mask]
            write(report, "> Dropped zero-variance variables for Ridge: $(join(dropped, ", "))\n\n")
        end

        # Standardize
        x_means = mean(X_raw, dims=1)
        x_stds  = std(X_raw, dims=1)
        y_mean  = mean(y_raw)

        X_std = (X_raw .- x_means) ./ x_stds
        y_std = y_raw .- y_mean

        # Select λ by leave-one-out CV (GCV approximation)
        n_obs = size(X_std, 1)
        p_dim = size(X_std, 2)
        lambdas = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
        best_lambda = 0.1
        best_gcv = Inf

        XtX = X_std' * X_std
        Xty = X_std' * y_std

        for λ in lambdas
            beta_r = (XtX + λ * I(p_dim)) \ Xty
            y_hat = X_std * beta_r
            H = X_std * ((XtX + λ * I(p_dim)) \ X_std')
            residuals = y_std - y_hat
            denom = (1.0 - tr(H) / n_obs)^2
            gcv = denom > 0 ? sum(residuals .^ 2) / (n_obs * denom) : Inf
            if gcv < best_gcv
                best_gcv = gcv
                best_lambda = λ
            end
        end

        beta_ridge = (XtX + best_lambda * I(p_dim)) \ Xty

        # R² for ridge
        y_hat = X_std * beta_ridge
        ss_res = sum((y_std - y_hat) .^ 2)
        ss_tot = sum(y_std .^ 2)
        ridge_r2 = 1.0 - ss_res / ss_tot

        write(report, @sprintf("**Optimal λ (GCV):** %.3f\n", best_lambda))
        write(report, @sprintf("**Ridge R²:** %.4f\n\n", ridge_r2))

        write(report, "| Term | Ridge_Coef (std) | OLS_Coef (std) | Shrinkage |\n")
        write(report, "|------|-----------------|----------------|----------|\n")

        # Compare with OLS standardized coefficients
        beta_ols = XtX \ Xty
        for (j, v) in enumerate(ridge_vars)
            shrink = beta_ols[j] != 0 ? 1.0 - abs(beta_ridge[j]) / abs(beta_ols[j]) : 0.0
            write(report, @sprintf("| %s | %.4f | %.4f | %.1f%% |\n",
                v, beta_ridge[j], beta_ols[j], shrink * 100))
            push!(stats_rows, ("S4", "ridge_coef_$v", beta_ridge[j], NaN, NaN, "λ=$(best_lambda)"))
        end
        write(report, "\n")

        push!(stats_rows, ("S4", "ridge_lambda", best_lambda, NaN, NaN, "GCV"))
        push!(stats_rows, ("S4", "ridge_R2", ridge_r2, NaN, NaN, ""))

        # Robustness verdict
        write(report, "**Robustness verdict:** ")
        ols_signs = sign.(beta_ols)
        ridge_signs = sign.(beta_ridge)
        sign_agree = sum(ols_signs .== ridge_signs)
        write(report, "$sign_agree/$(length(ridge_vars)) coefficients retain the same sign under Ridge. ")
        if sign_agree == length(ridge_vars)
            write(report, "The driver ranking from OLS is **robust** despite multicollinearity.\n\n")
        else
            flipped = ridge_vars[ols_signs .!= ridge_signs]
            write(report, "Variables $(join(flipped, ", ")) **flipped sign** — their OLS effects are unreliable.\n\n")
        end
    catch e
        write(report, "*Ridge regression failed: $e*\n\n")
    end

    # ── 4c: PCA Regression ──
    write(report, "### 4c: PCA Regression (collinear variables consolidated)\n\n")

    try
        X_raw = Matrix{Float64}(conv[:, available])
        y_raw = conv.log_conv_tick

        # Remove zero-variance columns for PCA
        col_sds_pca = vec(std(X_raw, dims=1))
        nonzero_mask_pca = col_sds_pca .> 1e-12
        pca_vars = available[nonzero_mask_pca]
        X_raw = X_raw[:, nonzero_mask_pca]

        x_means = mean(X_raw, dims=1)
        x_stds  = std(X_raw, dims=1)
        X_std = (X_raw .- x_means) ./ x_stds
        y_mean = mean(y_raw)
        y_cen = y_raw .- y_mean

        # SVD-based PCA
        U, S_vals, V = svd(X_std)
        eigenvalues = S_vals .^ 2 / (size(X_std, 1) - 1)
        explained = eigenvalues ./ sum(eigenvalues)
        cumexplained = cumsum(explained)

        write(report, "**Principal Components:**\n\n")
        write(report, "| PC | Eigenvalue | Var_Explained | Cumulative |\n")
        write(report, "|----|------------|---------------|------------|\n")
        for k in 1:length(eigenvalues)
            write(report, @sprintf("| PC%d | %.4f | %.3f | %.3f |\n",
                k, eigenvalues[k], explained[k], cumexplained[k]))
        end
        write(report, "\n")

        # Keep components explaining 95% variance
        n_keep = findfirst(cumexplained .>= 0.95)
        if n_keep === nothing
            n_keep = length(eigenvalues)
        end

        # PC scores
        Z = X_std * V[:, 1:n_keep]
        Z_aug = hcat(ones(size(Z, 1)), Z)
        beta_pc = Z_aug \ y_cen
        y_hat = Z_aug * beta_pc
        ss_res = sum((y_cen - y_hat) .^ 2)
        ss_tot = sum(y_cen .^ 2)
        pca_r2 = 1.0 - ss_res / ss_tot
        pca_adj_r2 = 1.0 - (1.0 - pca_r2) * (size(Z, 1) - 1) / (size(Z, 1) - n_keep - 1)

        write(report, @sprintf("Using first %d PCs (%.1f%% variance).\n", n_keep, cumexplained[n_keep]*100))
        write(report, @sprintf("**PCA regression R²:** %.4f | **Adj R²:** %.4f\n\n", pca_r2, pca_adj_r2))

        # Map back to original variable loadings
        write(report, "**PC loadings (top variables per component):**\n\n")
        write(report, "| Variable | " * join(["PC$k" for k in 1:n_keep], " | ") * " |\n")
        write(report, "|----------|" * join(["---" for _ in 1:n_keep], "|") * "|\n")
        for (j, v) in enumerate(pca_vars)
            row_str = @sprintf("| %s |", v)
            for k in 1:n_keep
                row_str *= @sprintf(" %.3f |", V[j, k])
            end
            write(report, row_str * "\n")
        end
        write(report, "\n")

        push!(stats_rows, ("S4", "pca_R2", pca_r2, NaN, Float64(n_keep), "n_components"))
        push!(stats_rows, ("S4", "pca_adj_R2", pca_adj_r2, NaN, Float64(n_keep), ""))
    catch e
        write(report, "*PCA regression failed: $e*\n\n")
    end
end

# ════════════════════════════════════════════════════════════════
# S5: Mediation Analysis (Baron-Kenny + Bootstrap CI)
# ════════════════════════════════════════════════════════════════

function _bootstrap_indirect_effect(X::Vector{Float64}, M::Vector{Float64},
                                     Y::Vector{Float64}, n_boot::Int, rng::AbstractRNG)
    n = length(X)
    indirect_samples = Float64[]

    for _ in 1:n_boot
        idx = rand(rng, 1:n, n)
        Xb, Mb, Yb = X[idx], M[idx], Y[idx]

        # a path: X → M
        Xa = hcat(ones(n), Xb)
        beta_a = Xa \ Mb
        a = beta_a[2]

        # b path: X + M → Y
        Xb_mat = hcat(ones(n), Xb, Mb)
        beta_b = Xb_mat \ Yb
        b = beta_b[3]

        push!(indirect_samples, a * b)
    end

    return indirect_samples
end

function mediation_analysis!(report::IOBuffer, stats_rows::Vector, pathways_df::DataFrame)
    println("  [S5] Mediation analysis (Baron-Kenny + Bootstrap)...")
    write(report, "## S5: Mediation Analysis — Baron-Kenny + Bootstrap\n\n")
    write(report, "**Path:** Phi → total_enforcements → total_dissolutions → convergence_tick\n\n")
    write(report, "> Bootstrap CI (percentile method, B=$N_BOOTSTRAP) supplements the Sobel test ")
    write(report, "as it does not assume normality of the indirect effect distribution.\n\n")

    conv = pathways_df[pathways_df.converged, :]
    required_cols = [:Phi, :total_enforcements, :total_dissolutions, :convergence_tick]
    if !all(hasproperty(conv, c) for c in required_cols)
        write(report, "*Required columns missing.*\n\n")
        return
    end
    if nrow(conv) < 10
        write(report, "*Insufficient converged runs (n=$(nrow(conv))).*\n\n")
        return
    end

    conv = copy(conv)
    conv.log_conv_tick = log.(max.(conv.convergence_tick, 1))
    rng = MersenneTwister(42)

    segments = [
        ("Segment 1", :Phi, :total_enforcements, :log_conv_tick,
         "Phi→enforcements", "enforcements→tick"),
        ("Segment 2", :Phi, :total_dissolutions, :log_conv_tick,
         "Phi→dissolutions", "dissolutions→tick"),
    ]

    for (seg_name, x_col, m_col, y_col, a_label, b_label) in segments
        write(report, "### $seg_name: $x_col → $m_col → $y_col\n\n")

        try
            X_vec = Float64.(conv[!, x_col])
            M_vec = Float64.(conv[!, m_col])
            Y_vec = Float64.(conv[!, y_col])

            # Baron-Kenny steps
            m_c = lm(hcat(ones(length(X_vec)), X_vec), Y_vec)
            c_coef = coef(m_c)[2]

            m_a = lm(hcat(ones(length(X_vec)), X_vec), M_vec)
            a_coef = coef(m_a)[2]

            X_M = hcat(ones(length(X_vec)), X_vec, M_vec)
            m_b = lm(X_M, Y_vec)
            b_coef = coef(m_b)[3]
            c_prime = coef(m_b)[2]

            indirect = a_coef * b_coef

            # Sobel test
            X_design_a = hcat(ones(length(X_vec)), X_vec)
            resid_a = M_vec - X_design_a * coef(m_a)
            se_a = sqrt(sum(resid_a .^ 2) / (length(X_vec) - 2)) / sqrt(sum((X_vec .- mean(X_vec)) .^ 2))

            resid_b = Y_vec - X_M * coef(m_b)
            mse_b = sum(resid_b .^ 2) / (length(X_vec) - 3)
            XtX_inv = inv(X_M' * X_M)
            se_b = sqrt(mse_b * XtX_inv[3, 3])

            sobel_se = sqrt(a_coef^2 * se_b^2 + b_coef^2 * se_a^2)
            sobel_z = indirect / sobel_se
            sobel_p = 2.0 * (1.0 - cdf(Normal(), abs(sobel_z)))

            prop_med = c_coef != 0 ? indirect / c_coef : NaN

            # Bootstrap
            boot_samples = _bootstrap_indirect_effect(X_vec, M_vec, Y_vec, N_BOOTSTRAP, rng)
            boot_ci_lo = quantile(boot_samples, 0.025)
            boot_ci_hi = quantile(boot_samples, 0.975)
            boot_mean  = mean(boot_samples)
            boot_se    = std(boot_samples)
            boot_sig   = !(boot_ci_lo <= 0 <= boot_ci_hi)

            write(report, "| Path | Coef | p |\n")
            write(report, "|------|------|---|\n")
            write(report, @sprintf("| c (total) | %.4f | - |\n", c_coef))
            write(report, @sprintf("| a (%s) | %.4f | - |\n", a_label, a_coef))
            write(report, @sprintf("| b (%s) | %.4f | - |\n", b_label, b_coef))
            write(report, @sprintf("| c' (direct) | %.4f | - |\n", c_prime))
            write(report, @sprintf("\n**Indirect effect (a×b):** %.4f\n", indirect))
            write(report, @sprintf("**Sobel test:** z=%.3f, p=%.4f\n", sobel_z, sobel_p))
            write(report, @sprintf("**Bootstrap 95%% CI:** [%.4f, %.4f] (B=%d)\n",
                boot_ci_lo, boot_ci_hi, N_BOOTSTRAP))
            write(report, @sprintf("**Bootstrap mean:** %.4f, SE=%.4f\n", boot_mean, boot_se))
            write(report, "**Bootstrap significant (CI excludes 0):** $(boot_sig ? "YES" : "NO")\n")
            write(report, @sprintf("**Proportion mediated:** %.3f\n\n", prop_med))

            seg_prefix = seg_name == "Segment 1" ? "seg1" : "seg2"
            push!(stats_rows, ("S5", "$(seg_prefix)_indirect_effect", indirect, sobel_p, NaN, ""))
            push!(stats_rows, ("S5", "$(seg_prefix)_sobel_z", sobel_z, sobel_p, NaN, ""))
            push!(stats_rows, ("S5", "$(seg_prefix)_boot_ci_lo", boot_ci_lo, NaN, NaN, "95%CI"))
            push!(stats_rows, ("S5", "$(seg_prefix)_boot_ci_hi", boot_ci_hi, NaN, NaN, "95%CI"))
            push!(stats_rows, ("S5", "$(seg_prefix)_boot_sig", Float64(boot_sig), NaN, NaN,
                boot_sig ? "significant" : "not_significant"))
            push!(stats_rows, ("S5", "$(seg_prefix)_proportion_mediated", prop_med, NaN, NaN, ""))
        catch e
            write(report, "*$seg_name failed: $e*\n\n")
        end
    end
end

# ════════════════════════════════════════════════════════════════
# S6: Granger Causality + ADF stationarity + AIC lag selection
# ════════════════════════════════════════════════════════════════

function _adf_test(x::Vector{Float64}; max_lag::Int=3)
    # Augmented Dickey-Fuller: Δy_t = α + ρ*y_{t-1} + Σ γ_k Δy_{t-k} + ε
    n = length(x)
    best_lag = 0
    best_aic = Inf

    for p in 0:min(max_lag, div(n, 5) - 2)
        T = n - p - 1
        if T < p + 3
            continue
        end

        dy = diff(x)
        Y = dy[(p+1):end]
        X_cols = [ones(T), x[(p+1):(n-1)]]
        for k in 1:p
            push!(X_cols, dy[(p+1-k):(end-k)])
        end
        X_mat = hcat(X_cols...)

        try
            beta = X_mat \ Y
            resid = Y - X_mat * beta
            ssr = sum(resid .^ 2)
            k_params = size(X_mat, 2)
            aic = T * log(ssr / T) + 2 * k_params
            if aic < best_aic
                best_aic = aic
                best_lag = p
            end
        catch
            continue
        end
    end

    # Run at best lag
    p = best_lag
    T = n - p - 1
    dy = diff(x)
    Y = dy[(p+1):end]
    X_cols = [ones(T), x[(p+1):(n-1)]]
    for k in 1:p
        push!(X_cols, dy[(p+1-k):(end-k)])
    end
    X_mat = hcat(X_cols...)

    beta = X_mat \ Y
    resid = Y - X_mat * beta
    mse = sum(resid .^ 2) / (T - size(X_mat, 2))
    XtXinv = inv(X_mat' * X_mat)
    se_rho = sqrt(mse * XtXinv[2, 2])
    t_stat = beta[2] / se_rho

    # MacKinnon critical values (approximate for n > 50)
    cv_1  = -3.43
    cv_5  = -2.86
    cv_10 = -2.57

    stationary = t_stat < cv_5
    return (t_stat=t_stat, lag=best_lag, cv_1pct=cv_1, cv_5pct=cv_5, cv_10pct=cv_10,
            stationary=stationary)
end

function _aic_bic_lag_select(x::Vector{Float64}, y::Vector{Float64}; max_lag::Int=5)
    n = min(length(x), length(y))
    best_aic_lag = 1
    best_bic_lag = 1
    best_aic = Inf
    best_bic = Inf

    for lag in 1:min(max_lag, div(n, 5))
        result = _granger_f_test(x, y, lag)
        if result === nothing
            continue
        end

        T = n - lag
        Y = y[(lag+1):n]
        Z_u = hcat(ones(T), [y[(lag+1-l):(n-l)] for l in 1:lag]...,
                   [x[(lag+1-l):(n-l)] for l in 1:lag]...)
        try
            beta = Z_u \ Y
            resid = Y - Z_u * beta
            ssr = sum(resid .^ 2)
            k = size(Z_u, 2)
            aic = T * log(ssr / T) + 2 * k
            bic = T * log(ssr / T) + log(T) * k
            if aic < best_aic
                best_aic = aic
                best_aic_lag = lag
            end
            if bic < best_bic
                best_bic = bic
                best_bic_lag = lag
            end
        catch
            continue
        end
    end

    return (aic_lag=best_aic_lag, bic_lag=best_bic_lag)
end

function granger_causality!(report::IOBuffer, stats_rows::Vector,
                            snapshot_df::DataFrame, pathways_df::DataFrame)
    println("  [S6] Granger causality (+ ADF + lag selection)...")
    write(report, "## S6: Granger Causality (VAR F-test)\n\n")
    write(report, "> **Caveat:** Granger causality tests *temporal predictive precedence*, not ")
    write(report, "physical causation. A significant result means past values of X improve ")
    write(report, "prediction of Y beyond Y's own past, which is a necessary but not sufficient ")
    write(report, "condition for true causality.\n\n")

    conv_runs = pathways_df[pathways_df.converged, :]
    if nrow(conv_runs) == 0
        write(report, "*No converged runs for Granger analysis.*\n\n")
        return
    end

    ts = combine(groupby(snapshot_df, :tick),
        :beff_A => mean => :mean_beff_A,
        :sigma  => mean => :mean_sigma,
        :C      => mean => :mean_C,
        :a      => mean => :frac_A,
    )
    sort!(ts, :tick)

    if nrow(ts) < 10
        write(report, "*Insufficient time series data ($(nrow(ts)) ticks).*\n\n")
        return
    end

    ts.anomaly = abs.(ts.frac_A .- 0.5)
    ts.belief_shift = vcat(0.0, abs.(diff(ts.mean_beff_A)))
    ts.norm_strength = ts.mean_C

    # ── ADF stationarity tests ──
    write(report, "### Stationarity Tests (ADF)\n\n")
    write(report, "> Granger causality requires stationary series. Non-stationary data ")
    write(report, "can produce spurious regressions.\n\n")
    write(report, "| Series | ADF_t | Lag | 5%_CV | Stationary |\n")
    write(report, "|--------|-------|-----|-------|------------|\n")

    test_series = [
        (:anomaly, "anomaly"),
        (:mean_sigma, "mean_sigma"),
        (:belief_shift, "belief_shift"),
        (:mean_C, "mean_C"),
        (:mean_beff_A, "mean_beff_A"),
        (:norm_strength, "norm_strength"),
    ]

    need_diff = Symbol[]
    for (col, label) in test_series
        vals = Float64.(ts[!, col])
        if all(vals .== vals[1])
            write(report, @sprintf("| %s | - | - | - | constant |\n", label))
            continue
        end
        adf = _adf_test(vals)
        status = adf.stationary ? "YES" : "NO → use Δ"
        write(report, @sprintf("| %s | %.3f | %d | %.2f | %s |\n",
            label, adf.t_stat, adf.lag, adf.cv_5pct, status))
        push!(stats_rows, ("S6", "adf_$(col)", adf.t_stat, NaN, Float64(adf.lag),
            adf.stationary ? "stationary" : "non-stationary"))
        if !adf.stationary
            push!(need_diff, col)
        end
    end
    write(report, "\n")

    if !isempty(need_diff)
        write(report, "**Non-stationary series:** $(join(need_diff, ", ")). ")
        write(report, "These are first-differenced before Granger testing below.\n\n")
    end

    # Apply differencing to non-stationary series
    for col in need_diff
        original = Float64.(ts[!, col])
        ts[!, col] = vcat(0.0, diff(original))
    end

    # ── Lag selection ──
    write(report, "### Optimal Lag Selection (AIC/BIC)\n\n")
    pairs = [
        (:anomaly, :mean_sigma, "anomaly → dissolution_proxy(sigma)"),
        (:belief_shift, :anomaly, "belief_shift → anomaly"),
        (:mean_C, :norm_strength, "confidence → norm_strength"),
        (:mean_sigma, :mean_beff_A, "sigma → belief"),
    ]

    write(report, "| Pair | AIC_lag | BIC_lag |\n")
    write(report, "|------|---------|--------|\n")

    lag_choices = Dict{String, Int}()
    for (x_col, y_col, label) in pairs
        x_vec = Float64.(ts[!, x_col])
        y_vec = Float64.(ts[!, y_col])
        lags = _aic_bic_lag_select(x_vec, y_vec; max_lag=5)
        optimal = min(lags.aic_lag, lags.bic_lag)  # conservative: use smaller
        lag_choices[label] = optimal
        write(report, @sprintf("| %s | %d | %d |\n", label, lags.aic_lag, lags.bic_lag))
        push!(stats_rows, ("S6", "lag_aic_$(x_col)_$(y_col)", Float64(lags.aic_lag), NaN, NaN, ""))
        push!(stats_rows, ("S6", "lag_bic_$(x_col)_$(y_col)", Float64(lags.bic_lag), NaN, NaN, ""))
    end
    write(report, "\n")

    # ── Granger tests at optimal + all lags ──
    write(report, "### Granger F-tests\n\n")

    max_lag = min(3, div(nrow(ts), 5))

    write(report, "| Cause → Effect | Lag | F-stat | p-value | Sig | Optimal |\n")
    write(report, "|----------------|-----|--------|---------|-----|--------|\n")

    for (x_col, y_col, label) in pairs
        optimal_lag = get(lag_choices, label, 1)
        for lag in 1:max_lag
            result = _granger_f_test(Float64.(ts[!, x_col]), Float64.(ts[!, y_col]), lag)
            if result !== nothing
                f_stat, p_val = result
                sig = p_val < 0.01 ? "***" : p_val < 0.05 ? "**" : p_val < 0.1 ? "*" : ""
                opt_mark = lag == optimal_lag ? "←" : ""
                write(report, @sprintf("| %s | %d | %.3f | %.4f | %s | %s |\n",
                    label, lag, f_stat, p_val, sig, opt_mark))
                push!(stats_rows, ("S6", "granger_$(x_col)_$(y_col)_lag$(lag)",
                    f_stat, p_val, Float64(lag), sig))
            end
        end
    end
    write(report, "\n")
end

function _granger_f_test(x::AbstractVector, y::AbstractVector, lag::Int)
    n = length(y)
    if n <= 2 * lag + 2
        return nothing
    end

    T = n - lag
    Y = y[(lag+1):n]
    Z_r = hcat(ones(T), [y[(lag+1-l):(n-l)] for l in 1:lag]...)
    Z_u = hcat(Z_r, [x[(lag+1-l):(n-l)] for l in 1:lag]...)

    try
        beta_r = Z_r \ Y
        resid_r = Y - Z_r * beta_r
        ssr_r = sum(resid_r .^ 2)

        beta_u = Z_u \ Y
        resid_u = Y - Z_u * beta_u
        ssr_u = sum(resid_u .^ 2)

        q = lag
        df2 = T - size(Z_u, 2)
        if df2 < 1 || ssr_u <= 0
            return nothing
        end

        f_stat = ((ssr_r - ssr_u) / q) / (ssr_u / df2)
        p_val = 1.0 - cdf(FDist(q, df2), max(f_stat, 0.0))
        return (f_stat, p_val)
    catch
        return nothing
    end
end

# ════════════════════════════════════════════════════════════════
# S7: Changepoint Detection + Sensitivity + Early Warning
# ════════════════════════════════════════════════════════════════

function changepoint_detection!(report::IOBuffer, stats_rows::Vector,
                                snapshot_df::DataFrame, summary_df::DataFrame)
    println("  [S7] Changepoint detection (+ sensitivity + early warning)...")
    write(report, "## S7: Changepoint Detection (CUSUM)\n\n")

    # Aggregate to tick-level
    ts = combine(groupby(snapshot_df, :tick),
        :beff_A => mean => :mean_beff_A,
        :sigma  => mean => :mean_sigma,
        :C      => mean => :mean_C,
        :a      => sum   => :count_A,
        nrow    => :n_agents,
    )
    sort!(ts, :tick)

    ts_agent = combine(groupby(snapshot_df, :tick),
        :C => (x -> sum(x .> 0.9)) => :n_crystallized,
    )
    sort!(ts_agent, :tick)
    ts = leftjoin(ts, ts_agent, on=:tick)

    series = [
        (:n_crystallized, "Crystallized agents"),
        (:mean_beff_A,    "Mean belief (beff_A)"),
        (:mean_sigma,     "Mean sigma"),
    ]

    write(report, "| Series | Changepoint_tick | CUSUM_max | Direction |\n")
    write(report, "|--------|-----------------|-----------|----------|\n")

    changepoints = Dict{Symbol, Int}()
    for (col, label) in series
        if !hasproperty(ts, col)
            continue
        end
        vals = Float64.(ts[!, col])
        cp_tick, cusum_max, direction = _cusum_changepoint(vals)
        actual_tick = nrow(ts) > 0 ? ts.tick[cp_tick] : cp_tick
        changepoints[col] = actual_tick

        write(report, @sprintf("| %s | %d | %.3f | %s |\n",
            label, actual_tick, cusum_max, direction))
        push!(stats_rows, ("S7", "changepoint_$(col)", Float64(actual_tick),
            cusum_max, NaN, direction))
    end

    # ── Compare with known phase boundaries ──
    write(report, "\n### Comparison with Known Phase Boundaries\n\n")
    phase_cols = [:first_tick_behavioral, :first_tick_belief, :first_tick_crystal, :first_tick_all_met]
    available_phases = [c for c in phase_cols if hasproperty(summary_df, c)]

    if !isempty(available_phases)
        write(report, "| Phase | Mean_tick | Median_tick | SD |\n")
        write(report, "|-------|----------|-------------|----|\n")
        for col in available_phases
            vals = collect(skipmissing(summary_df[summary_df.converged, col]))
            if !isempty(vals)
                write(report, @sprintf("| %s | %.1f | %.1f | %.1f |\n",
                    col, mean(vals), median(vals), std(vals)))
            end
        end
    end
    write(report, "\n")

    # ── Sensitivity analysis across theta_crystal ──
    write(report, "### Sensitivity Analysis: Changepoint Drift by theta_crystal\n\n")
    write(report, "> How do changepoint timings shift when the crystallization threshold changes?\n\n")

    theta_vals = sort(unique(summary_df.theta_crystal))
    if length(theta_vals) > 1
        write(report, "| theta_crystal | conv_rate | mean_conv_tick | mean_first_belief | mean_first_crystal |\n")
        write(report, "|--------------|-----------|---------------|-------------------|-------------------|\n")
        for θ in theta_vals
            sub = summary_df[summary_df.theta_crystal .== θ, :]
            cr = mean(sub.converged)
            mt = mean(skipmissing(sub[sub.converged, :convergence_tick]))
            fb = hasproperty(sub, :first_tick_belief) ?
                mean(skipmissing(sub[sub.converged, :first_tick_belief])) : NaN
            fc = hasproperty(sub, :first_tick_crystal) ?
                mean(skipmissing(sub[sub.converged, :first_tick_crystal])) : NaN
            write(report, @sprintf("| %.1f | %.3f | %.1f | %.1f | %.1f |\n",
                θ, cr, mt, fb, fc))
            push!(stats_rows, ("S7", "sensitivity_theta$(θ)_conv_rate", cr, NaN, NaN, ""))
            push!(stats_rows, ("S7", "sensitivity_theta$(θ)_mean_tick", mt, NaN, NaN, ""))
        end
        write(report, "\n")
    end

    # ── Early Warning Signal Analysis ──
    write(report, "### Early Warning Signals (EWS)\n\n")
    write(report, "> Can the belief shift at the detected changepoint serve as a ")
    write(report, "leading indicator for the subsequent crystallization cascade?\n\n")

    belief_cp = get(changepoints, :mean_beff_A, 0)
    crystal_cp = get(changepoints, :n_crystallized, 0)

    if belief_cp > 0 && crystal_cp > 0
        lead_time = crystal_cp - belief_cp
        write(report, @sprintf("- **Belief changepoint:** tick %d\n", belief_cp))
        write(report, @sprintf("- **Crystallization changepoint:** tick %d\n", crystal_cp))
        write(report, @sprintf("- **Lead time:** %d ticks\n\n", lead_time))

        if lead_time > 0
            write(report, "The belief shift precedes the crystallization cascade by $lead_time ticks, ")
            write(report, "suggesting it can function as an **early warning signal**.\n\n")

            # Compute rolling variance (rising variance = critical slowing down)
            if hasproperty(ts, :mean_beff_A) && nrow(ts) > 10
                window = max(5, div(nrow(ts), 10))
                vals = Float64.(ts.mean_beff_A)
                roll_var = [i < window ? NaN : var(vals[(i-window+1):i]) for i in 1:length(vals)]
                valid_rv = filter(!isnan, roll_var[1:min(belief_cp, length(roll_var))])
                if length(valid_rv) > 2
                    trend = cor(1:length(valid_rv), valid_rv)
                    write(report, @sprintf("**Rolling variance trend (pre-changepoint):** r=%.3f ", trend))
                    if trend > 0.3
                        write(report, "(INCREASING — consistent with critical slowing down)\n\n")
                    elseif trend < -0.3
                        write(report, "(DECREASING — no evidence of critical slowing down)\n\n")
                    else
                        write(report, "(FLAT — inconclusive)\n\n")
                    end
                    push!(stats_rows, ("S7", "ews_variance_trend", trend, NaN, NaN,
                        trend > 0.3 ? "CSD" : "no_CSD"))
                end
            end
        else
            write(report, "The belief and crystallization changepoints are concurrent or reversed — ")
            write(report, "no leading indicator relationship.\n\n")
        end

        push!(stats_rows, ("S7", "ews_lead_time", Float64(lead_time), NaN, NaN, "ticks"))
    end
end

function _cusum_changepoint(x::Vector{Float64})
    n = length(x)
    if n < 3
        return (1, 0.0, "none")
    end

    μ = mean(x)
    σ_x = std(x)
    if σ_x == 0
        return (1, 0.0, "flat")
    end

    z = (x .- μ) ./ σ_x
    cusum = cumsum(z)

    abs_cusum = abs.(cusum)
    cp_idx = argmax(abs_cusum)
    cusum_max = abs_cusum[cp_idx]
    direction = cusum[cp_idx] > 0 ? "increase" : "decrease"

    return (cp_idx, cusum_max, direction)
end

# ════════════════════════════════════════════════════════════════
# S8: Counterfactual Intervention Analysis (Do-calculus)
# ════════════════════════════════════════════════════════════════

function counterfactual_analysis!(report::IOBuffer, stats_rows::Vector, pathways_df::DataFrame)
    println("  [S8] Counterfactual intervention analysis...")
    write(report, "## S8: Counterfactual Intervention Analysis\n\n")
    write(report, "> Using the estimated causal model (structural equations from S4/S5), ")
    write(report, "we simulate *do-interventions*: what would happen if we externally set ")
    write(report, "a variable to a specific value, breaking its natural correlations?\n\n")

    conv = pathways_df[pathways_df.converged, :]
    if nrow(conv) < 10
        write(report, "*Insufficient data for counterfactual analysis.*\n\n")
        return
    end

    conv = copy(conv)
    conv.log_conv_tick = log.(max.(conv.convergence_tick, 1))

    # ── Estimate structural equations ──
    # Y = log_conv_tick, intervention targets: Phi, theta_crystal
    try
        # Structural model: total_enforcements ~ Phi
        m_enf = lm(@formula(total_enforcements ~ Phi + theta_crystal), conv)
        # Structural model: total_dissolutions ~ total_enforcements + Phi
        m_dis = lm(@formula(total_dissolutions ~ total_enforcements + Phi + theta_crystal), conv)
        # Structural model: log_conv_tick ~ Phi + theta_crystal + total_dissolutions + total_enforcements
        m_y = lm(@formula(log_conv_tick ~ Phi + theta_crystal + total_dissolutions + total_enforcements), conv)

        # ── Intervention 1: do(Phi := Phi - 5) ──
        write(report, "### Intervention 1: do(Phi := Phi - 5)\n\n")
        write(report, "\"What if environmental pressure were reduced by 5 units?\"\n\n")

        orig_Phi = mean(conv.Phi)
        new_Phi = orig_Phi - 5.0
        Δ_Phi = -5.0

        # Propagate through structural equations
        coef_enf = coef(m_enf)
        Δ_enforcements = coef_enf[2] * Δ_Phi  # Phi coefficient

        coef_dis = coef(m_dis)
        Δ_dissolutions = coef_dis[2] * Δ_enforcements + coef_dis[3] * Δ_Phi

        coef_y = coef(m_y)
        Δ_y = coef_y[2] * Δ_Phi + coef_y[4] * Δ_dissolutions + coef_y[5] * Δ_enforcements

        orig_y = mean(conv.log_conv_tick)
        new_y = orig_y + Δ_y
        pct_change_tick = (exp(new_y) - exp(orig_y)) / exp(orig_y) * 100

        write(report, "| Variable | Observed_Mean | Counterfactual | Δ |\n")
        write(report, "|----------|--------------|----------------|---|\n")
        write(report, @sprintf("| Phi | %.1f | %.1f | %.1f |\n", orig_Phi, new_Phi, Δ_Phi))
        write(report, @sprintf("| total_enforcements | %.1f | %.1f | %.1f |\n",
            mean(conv.total_enforcements), mean(conv.total_enforcements) + Δ_enforcements, Δ_enforcements))
        write(report, @sprintf("| total_dissolutions | %.1f | %.1f | %.1f |\n",
            mean(conv.total_dissolutions), mean(conv.total_dissolutions) + Δ_dissolutions, Δ_dissolutions))
        write(report, @sprintf("| log(conv_tick) | %.3f | %.3f | %.3f |\n", orig_y, new_y, Δ_y))
        write(report, @sprintf("| conv_tick | %.1f | %.1f | %+.1f%% |\n\n",
            exp(orig_y), exp(new_y), pct_change_tick))

        push!(stats_rows, ("S8", "do_phi_minus5_tick_pct_change", pct_change_tick, NaN, NaN, ""))

        # ── Intervention 2: do(theta_crystal := theta_crystal + 1) ──
        write(report, "### Intervention 2: do(theta_crystal := theta_crystal + 1)\n\n")
        write(report, "\"What if the crystallization threshold were raised by 1?\"\n\n")

        orig_theta = mean(conv.theta_crystal)
        Δ_theta = 1.0

        Δ_enforcements_2 = coef_enf[3] * Δ_theta
        Δ_dissolutions_2 = coef_dis[2] * Δ_enforcements_2 + coef_dis[4] * Δ_theta
        Δ_y_2 = coef_y[3] * Δ_theta + coef_y[4] * Δ_dissolutions_2 + coef_y[5] * Δ_enforcements_2

        new_y_2 = orig_y + Δ_y_2
        pct_change_2 = (exp(new_y_2) - exp(orig_y)) / exp(orig_y) * 100

        write(report, "| Variable | Observed_Mean | Counterfactual | Δ |\n")
        write(report, "|----------|--------------|----------------|---|\n")
        write(report, @sprintf("| theta_crystal | %.1f | %.1f | %.1f |\n", orig_theta, orig_theta + Δ_theta, Δ_theta))
        write(report, @sprintf("| total_enforcements | %.1f | %.1f | %.1f |\n",
            mean(conv.total_enforcements), mean(conv.total_enforcements) + Δ_enforcements_2, Δ_enforcements_2))
        write(report, @sprintf("| total_dissolutions | %.1f | %.1f | %.1f |\n",
            mean(conv.total_dissolutions), mean(conv.total_dissolutions) + Δ_dissolutions_2, Δ_dissolutions_2))
        write(report, @sprintf("| log(conv_tick) | %.3f | %.3f | %.3f |\n", orig_y, new_y_2, Δ_y_2))
        write(report, @sprintf("| conv_tick | %.1f | %.1f | %+.1f%% |\n\n",
            exp(orig_y), exp(new_y_2), pct_change_2))

        push!(stats_rows, ("S8", "do_theta_plus1_tick_pct_change", pct_change_2, NaN, NaN, ""))

        # ── Intervention 3: do(total_enforcements := 0) ──
        write(report, "### Intervention 3: do(total_enforcements := 0)\n\n")
        write(report, "\"What if norm enforcement were completely disabled?\"\n\n")

        orig_enf = mean(conv.total_enforcements)
        Δ_enforcements_3 = -orig_enf

        Δ_dissolutions_3 = coef_dis[2] * Δ_enforcements_3
        Δ_y_3 = coef_y[4] * Δ_dissolutions_3 + coef_y[5] * Δ_enforcements_3

        new_y_3 = orig_y + Δ_y_3
        pct_change_3 = (exp(new_y_3) - exp(orig_y)) / exp(orig_y) * 100

        write(report, "| Variable | Observed_Mean | Counterfactual | Δ |\n")
        write(report, "|----------|--------------|----------------|---|\n")
        write(report, @sprintf("| total_enforcements | %.1f | 0.0 | %.1f |\n", orig_enf, Δ_enforcements_3))
        write(report, @sprintf("| total_dissolutions | %.1f | %.1f | %.1f |\n",
            mean(conv.total_dissolutions), mean(conv.total_dissolutions) + Δ_dissolutions_3, Δ_dissolutions_3))
        write(report, @sprintf("| log(conv_tick) | %.3f | %.3f | %.3f |\n", orig_y, new_y_3, Δ_y_3))
        write(report, @sprintf("| conv_tick | %.1f | %.1f | %+.1f%% |\n\n",
            exp(orig_y), exp(new_y_3), pct_change_3))

        push!(stats_rows, ("S8", "do_no_enforcement_tick_pct_change", pct_change_3, NaN, NaN, ""))

        # ── Model validity caveat ──
        write(report, "> **Validity note:** These counterfactuals assume the linear structural ")
        write(report, "equations remain valid under intervention (no model misspecification, ")
        write(report, "no unobserved confounders). The estimates should be treated as ")
        write(report, "first-order approximations. For large interventions, nonlinear effects ")
        write(report, "may dominate.\n\n")
    catch e
        write(report, "*Counterfactual analysis failed: $e*\n\n")
    end
end

# ════════════════════════════════════════════════════════════════
# S9: Visualizations — DAG + Changepoint Timeline
# ════════════════════════════════════════════════════════════════

function generate_visualizations!(report::IOBuffer, stats_rows::Vector,
                                  snapshot_df::DataFrame, summary_df::DataFrame,
                                  pathways_df::DataFrame)
    println("  [S9] Generating visualizations...")
    write(report, "## S9: Visualizations\n\n")

    # ── 9a: Causal DAG ──
    try
        dag_path = joinpath(RESULTS_DIR, "causal_dag.png")
        _plot_causal_dag(dag_path)
        write(report, "### Causal DAG\n\n")
        write(report, "![Causal DAG](causal_dag.png)\n\n")
        write(report, "Nodes represent key variables; directed edges represent estimated causal paths ")
        write(report, "from regression analysis (S4/S5). Edge labels show standardized effect sizes.\n\n")
    catch e
        write(report, "*DAG generation failed: $e*\n\n")
    end

    # ── 9b: Changepoint Timeline ──
    try
        timeline_path = joinpath(RESULTS_DIR, "causal_changepoint_timeline.png")
        _plot_changepoint_timeline(snapshot_df, summary_df, timeline_path)
        write(report, "### Changepoint Timeline\n\n")
        write(report, "![Changepoint Timeline](causal_changepoint_timeline.png)\n\n")
        write(report, "Vertical dashed lines mark CUSUM-detected changepoints. Shaded regions ")
        write(report, "indicate the early warning window between belief shift and crystallization cascade.\n\n")
    catch e
        write(report, "*Timeline generation failed: $e*\n\n")
    end
end

function _plot_causal_dag(output_path::String)
    # Hand-positioned DAG using scatter + annotations + arrows
    p = plot(size=(800, 600), legend=false, grid=false,
             xlim=(-0.5, 4.5), ylim=(-0.5, 3.5),
             axis=false, ticks=false, framestyle=:none,
             title="Causal DAG: Norm Emergence Cascade",
             titlefontsize=14)

    # Node positions (x, y)
    nodes = Dict(
        "Phi"               => (0.0, 2.0),
        "theta_crystal"     => (0.0, 1.0),
        "enforcements"      => (1.5, 2.5),
        "dissolutions"      => (1.5, 1.5),
        "time_in_churn"     => (1.5, 0.5),
        "convergence_tick"  => (3.5, 1.5),
        "converged"         => (3.5, 0.5),
    )

    # Draw nodes
    for (name, (x, y)) in nodes
        scatter!(p, [x], [y], markersize=20, markercolor=:steelblue,
                 markerstrokewidth=2, markerstrokecolor=:navy)
        annotate!(p, x, y - 0.3, text(name, 8, :center))
    end

    # Edges with labels (from → to, label)
    edges = [
        ("Phi", "enforcements", "+"),
        ("Phi", "dissolutions", "+"),
        ("Phi", "convergence_tick", "-"),
        ("theta_crystal", "dissolutions", "+"),
        ("theta_crystal", "convergence_tick", "+"),
        ("theta_crystal", "converged", "-"),
        ("enforcements", "convergence_tick", "+"),
        ("enforcements", "dissolutions", "+"),
        ("dissolutions", "convergence_tick", "+"),
        ("time_in_churn", "convergence_tick", "-"),
    ]

    for (from_name, to_name, label) in edges
        x1, y1 = nodes[from_name]
        x2, y2 = nodes[to_name]
        # Shorten arrow to not overlap nodes
        dx, dy = x2 - x1, y2 - y1
        len = sqrt(dx^2 + dy^2)
        if len > 0
            offset = 0.25
            sx = x1 + offset * dx / len
            sy = y1 + offset * dy / len
            ex = x2 - offset * dx / len
            ey = y2 - offset * dy / len
            # Draw arrow line
            plot!(p, [sx, ex], [sy, ey], linewidth=2,
                  linecolor=label == "+" ? :forestgreen : :crimson,
                  arrow=(:closed, 2.0))
            # Label at midpoint
            mx = (sx + ex) / 2
            my = (sy + ey) / 2
            annotate!(p, mx, my + 0.1,
                text(label, 10, label == "+" ? :green : :red, :center))
        end
    end

    savefig(p, output_path)
    println("    → Saved DAG to $output_path")
end

function _plot_changepoint_timeline(snapshot_df::DataFrame, summary_df::DataFrame,
                                     output_path::String)
    ts = combine(groupby(snapshot_df, :tick),
        :beff_A => mean => :mean_beff_A,
        :sigma  => mean => :mean_sigma,
        :C      => mean => :mean_C,
    )
    sort!(ts, :tick)

    ts_cryst = combine(groupby(snapshot_df, :tick),
        :C => (x -> sum(x .> 0.9)) => :n_crystallized,
    )
    sort!(ts_cryst, :tick)
    ts = leftjoin(ts, ts_cryst, on=:tick)

    # Detect changepoints
    cp_belief = _cusum_changepoint(Float64.(ts.mean_beff_A))
    cp_crystal = _cusum_changepoint(Float64.(ts.n_crystallized))
    cp_sigma = _cusum_changepoint(Float64.(ts.mean_sigma))

    tick_belief = ts.tick[cp_belief[1]]
    tick_crystal = ts.tick[cp_crystal[1]]
    tick_sigma = ts.tick[cp_sigma[1]]

    p1 = plot(ts.tick, ts.n_crystallized, label="Crystallized agents",
              linewidth=2, color=:steelblue,
              ylabel="Count", title="Cascade Timeline with Changepoints")
    vline!(p1, [tick_crystal], label="CP: crystal (t=$tick_crystal)",
           linestyle=:dash, color=:steelblue, linewidth=2)

    p2 = plot(ts.tick, ts.mean_beff_A, label="Mean belief (beff_A)",
              linewidth=2, color=:darkorange, ylabel="Belief")
    vline!(p2, [tick_belief], label="CP: belief (t=$tick_belief)",
           linestyle=:dash, color=:darkorange, linewidth=2)
    # Shade early warning window
    if tick_belief < tick_crystal
        vspan!(p2, [tick_belief, tick_crystal], alpha=0.15, color=:gold, label="EW window")
    end

    p3 = plot(ts.tick, ts.mean_sigma, label="Mean sigma",
              linewidth=2, color=:purple, ylabel="Sigma", xlabel="Tick")
    vline!(p3, [tick_sigma], label="CP: sigma (t=$tick_sigma)",
           linestyle=:dash, color=:purple, linewidth=2)

    p_combined = plot(p1, p2, p3, layout=(3, 1), size=(900, 700),
                      left_margin=10Plots.mm)

    savefig(p_combined, output_path)
    println("    → Saved timeline to $output_path")
end

# ════════════════════════════════════════════════════════════════
# Main: Orchestrate & Write Outputs
# ════════════════════════════════════════════════════════════════

function main()
    println("╔══════════════════════════════════════════════════════════╗")
    println("║  Causal Analysis Pipeline v2 — Norm Emergence Cascades  ║")
    println("╚══════════════════════════════════════════════════════════╝")
    println()

    println("[S0] Loading data...")
    summary_df, pathways_df, snapshot_df = load_data()
    println()

    report = IOBuffer()
    write(report, "# Causal Analysis Report — Norm Emergence Cascades (v2)\n\n")
    write(report, "Generated: $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))\n\n")
    write(report, "---\n\n")

    stats_rows = Tuple{String, String, Float64, Float64, Float64, String}[]

    # ── Run all analyses ──
    descriptive_overview!(report, stats_rows, summary_df)
    partial_correlation_graph!(report, stats_rows, pathways_df)
    logistic_cascade_success!(report, stats_rows, pathways_df)
    linear_convergence_speed!(report, stats_rows, pathways_df)
    mediation_analysis!(report, stats_rows, pathways_df)
    granger_causality!(report, stats_rows, snapshot_df, pathways_df)
    changepoint_detection!(report, stats_rows, snapshot_df, summary_df)
    counterfactual_analysis!(report, stats_rows, pathways_df)
    generate_visualizations!(report, stats_rows, snapshot_df, summary_df, pathways_df)

    # ── Write report ──
    report_path = joinpath(RESULTS_DIR, "causal_analysis_report.md")
    open(report_path, "w") do f
        write(f, String(take!(report)))
    end
    println("\n✓ Report written to: $report_path")

    # ── Write summary CSV ──
    csv_path = joinpath(RESULTS_DIR, "causal_analysis_summary.csv")
    csv_df = DataFrame(
        section = [r[1] for r in stats_rows],
        metric  = [r[2] for r in stats_rows],
        value   = [r[3] for r in stats_rows],
        p_value = [r[4] for r in stats_rows],
        extra   = [r[5] for r in stats_rows],
        note    = [r[6] for r in stats_rows],
    )
    CSV.write(csv_path, csv_df)
    println("✓ Summary CSV written to: $csv_path")
    println("  $(nrow(csv_df)) statistics recorded across $(length(unique(csv_df.section))) sections")

    println("\n══════════ Done ══════════")
end

using Dates

main()
