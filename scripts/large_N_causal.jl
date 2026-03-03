#!/usr/bin/env julia
#
# Large-N Causal Analysis & Cross-N Comparison Report
#
# Two-phase operation:
#
# Phase A: Per-N causal analysis
#   - Runs a diagnostic trial at given N to produce agent snapshots
#   - Loads pathway data from large_N_cascade.jl A2 output
#   - Replicates causal_analysis.jl pipeline (S1-S7)
#   - Output: results/large_N/causal_N{N}_report.md
#             results/large_N/causal_N{N}_summary.csv
#
# Phase B: Cross-N comparison report
#   - Merges sweep + ablation + cascade results across all N
#   - Generates results/large_N/comparison_report.md
#
# Usage:
#   julia scripts/large_N_causal.jl N=1000     # Phase A only
#   julia scripts/large_N_causal.jl all         # Phase A (all N) + Phase B
#   julia scripts/large_N_causal.jl report      # Phase B only
#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DualMemoryABM
using CSV, DataFrames, Statistics, StatsBase
using GLM, HypothesisTests, Distributions, StatsModels
using LinearAlgebra
using Printf
using Random
using Dates

include(joinpath(@__DIR__, "large_N_common.jl"))

const OUTDIR = joinpath(@__DIR__, "..", "results", "large_N")
const N_ALL  = [1000, 2000, 5000, 10000, 20000]
const N_BOOTSTRAP = 2000
const DIAG_SEED = 42

# ══════════════════════════════════════════════════════════════
# Phase A: Per-N Causal Analysis
# ══════════════════════════════════════════════════════════════

# ── A.1: Generate diagnostic trial (agent snapshots) ──

function generate_snapshots(N::Int)
    progress_log("Generating diagnostic trial snapshots for N=$N...")
    T_MAX = select_T_MAX(N)

    params = SimulationParams(
        N = N, T = T_MAX, seed = DIAG_SEED,
        w_base = 2, w_max = 6,
        enable_normative = true,
        V = 0, Phi = 0.0,
        theta_crystal = 3.0,
    )

    agents, ws, history, rng = initialize(params)
    probes = ProbeSet()
    init_probes!(probes, params, rng)
    tick_count = 0

    snapshot_rows = []

    for t in 1:T_MAX
        run_tick!(t, agents, ws, history, tick_count, params, rng, probes)
        tick_count += 1

        # Record snapshot every 10 ticks (or every tick for first 100)
        if t <= 100 || t % 10 == 0
            for i in 1:N
                ag = agents[i]
                push!(snapshot_rows, (
                    tick    = t,
                    agent   = i,
                    r       = Int(ag.r),
                    bexp_A  = ag.b_exp_A,
                    beff_A  = ag.b_eff_A,
                    sigma   = ag.sigma,
                    a       = ag.a,
                    e       = ag.e,
                    C       = ag.C,
                ))
            end
        end

        if check_convergence(history, tick_count, params)
            break
        end
    end

    snapshot_path = joinpath(OUTDIR, "causal_snapshots_N$(N).csv")
    CSV.write(snapshot_path, snapshot_rows)
    progress_log("  → $(length(snapshot_rows)) snapshot rows saved")

    return snapshot_path, tick_count
end

# ── A.2: Granger F-test helper ──

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

# ── A.3: ADF stationarity test ──

function _adf_test(x::Vector{Float64}; max_lag::Int=3)
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

    cv_5 = -2.86
    stationary = t_stat < cv_5
    return (t_stat=t_stat, lag=best_lag, cv_5pct=cv_5, stationary=stationary)
end

# ── A.4: Per-N causal pipeline ──

function run_per_N_causal(N::Int)
    progress_log("Running per-N causal analysis for N=$N...")

    # Load pathway data
    pathway_path = joinpath(OUTDIR, "cascade_pathways_N$(N).csv")
    if !isfile(pathway_path)
        progress_log("  WARNING: $pathway_path not found, skipping")
        return
    end
    pathways_df = CSV.read(pathway_path, DataFrame)
    if eltype(pathways_df.converged) <: AbstractString
        pathways_df.converged = pathways_df.converged .== "true"
    end
    pathways_df.log_conv_tick = [
        row.converged ? log(max(row.convergence_tick, 1)) : missing
        for row in eachrow(pathways_df)
    ]

    # Generate or load snapshots
    snapshot_path = joinpath(OUTDIR, "causal_snapshots_N$(N).csv")
    if !isfile(snapshot_path)
        generate_snapshots(N)
    end
    snapshot_df = CSV.read(snapshot_path, DataFrame)

    progress_log("  Pathways: $(nrow(pathways_df)) rows, Snapshots: $(nrow(snapshot_df)) rows")

    # ── Build report ──
    report = IOBuffer()
    write(report, "# Causal Analysis Report — N=$N\n\n")
    write(report, "Generated: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))\n\n---\n\n")

    stats_rows = Tuple{String, String, Float64, Float64, Float64, String}[]

    # S1: Descriptive
    write(report, "## S1: Descriptive Overview\n\n")
    n_conv = sum(pathways_df.converged)
    conv_rate = mean(pathways_df.converged)
    write(report, @sprintf("**N=%d | Total runs:** %d | **Converged:** %d (%.1f%%)\n\n",
        N, nrow(pathways_df), n_conv, conv_rate * 100))
    push!(stats_rows, ("S1", "N", Float64(N), NaN, NaN, ""))
    push!(stats_rows, ("S1", "convergence_rate", conv_rate, NaN, NaN, ""))

    if n_conv > 0
        conv_ticks = pathways_df[pathways_df.converged, :convergence_tick]
        write(report, @sprintf("**Convergence tick:** mean=%.1f, median=%.1f, sd=%.1f\n\n",
            mean(conv_ticks), median(conv_ticks), std(conv_ticks)))
        push!(stats_rows, ("S1", "mean_conv_tick", mean(conv_ticks), NaN, NaN, ""))
    end

    gdf = groupby(pathways_df, [:theta_crystal, :Phi])
    desc = combine(gdf,
        :converged => mean => :conv_rate,
        :total_dissolutions => mean => :mean_diss,
        :total_enforcements => mean => :mean_enf,
        nrow => :n,
    )
    sort!(desc, [:theta_crystal, :Phi])
    write(report, "| θ | Φ | n | conv_rate | mean_diss | mean_enf |\n")
    write(report, "|---|---|---|-----------|-----------|----------|\n")
    for row in eachrow(desc)
        write(report, @sprintf("| %.1f | %.1f | %d | %.3f | %.1f | %.1f |\n",
            row.theta_crystal, row.Phi, row.n, row.conv_rate, row.mean_diss, row.mean_enf))
    end
    write(report, "\n")

    # S3: Logistic regression
    write(report, "## S3: Logistic Regression — Cascade Success\n\n")
    df_lr = copy(pathways_df)
    df_lr.converged_int = Int.(df_lr.converged)
    if length(unique(df_lr.converged_int)) >= 2
        try
            m1 = glm(@formula(converged_int ~ Phi + theta_crystal), df_lr, Binomial(), LogitLink())
            ct1 = coeftable(m1)
            write(report, "| Term | Coef | SE | z | p | OR |\n")
            write(report, "|------|------|----|---|---|----|\n")
            for i in 1:length(ct1.rownms)
                write(report, @sprintf("| %s | %.4f | %.4f | %.3f | %.4f | %.3f |\n",
                    ct1.rownms[i], ct1.cols[1][i], ct1.cols[2][i], ct1.cols[3][i],
                    ct1.cols[4][i], exp(ct1.cols[1][i])))
            end
            pseudo_r2 = 1.0 - loglikelihood(m1) / loglikelihood(
                glm(@formula(converged_int ~ 1), df_lr, Binomial(), LogitLink()))
            write(report, @sprintf("\n**Pseudo-R²:** %.4f\n\n", pseudo_r2))
            push!(stats_rows, ("S3", "pseudo_r2", pseudo_r2, NaN, NaN, ""))
        catch e
            write(report, "*Logistic regression failed: $e*\n\n")
        end
    else
        write(report, "*All runs have same outcome; skipping.*\n\n")
    end

    # S6: Granger causality
    write(report, "## S6: Granger Causality\n\n")
    ts = combine(groupby(snapshot_df, :tick),
        :beff_A => mean => :mean_beff_A,
        :sigma  => mean => :mean_sigma,
        :C      => mean => :mean_C,
        :a      => (x -> mean(x .> 0)) => :frac_anomaly,
    )
    sort!(ts, :tick)

    if nrow(ts) >= 20
        ts.anomaly = abs.(ts.frac_anomaly .- mean(ts.frac_anomaly))
        ts.belief_shift = vcat(0.0, abs.(diff(ts.mean_beff_A)))

        # ADF tests
        write(report, "### Stationarity (ADF)\n\n")
        write(report, "| Series | ADF_t | Stationary |\n")
        write(report, "|--------|-------|------------|\n")
        need_diff = Symbol[]
        for (col, label) in [(:anomaly, "anomaly"), (:mean_sigma, "mean_sigma"),
                              (:belief_shift, "belief_shift"), (:mean_C, "mean_C")]
            vals = Float64.(ts[!, col])
            if all(vals .== vals[1])
                write(report, @sprintf("| %s | - | constant |\n", label))
                continue
            end
            adf = _adf_test(vals)
            status = adf.stationary ? "YES" : "NO"
            write(report, @sprintf("| %s | %.3f | %s |\n", label, adf.t_stat, status))
            push!(stats_rows, ("S6", "adf_$(col)", adf.t_stat, NaN, NaN,
                adf.stationary ? "stationary" : "non-stationary"))
            if !adf.stationary
                push!(need_diff, col)
            end
        end
        write(report, "\n")

        for col in need_diff
            original = Float64.(ts[!, col])
            ts[!, col] = vcat(0.0, diff(original))
        end

        # Granger tests
        pairs = [
            (:anomaly, :mean_sigma, "anomaly→sigma"),
            (:belief_shift, :anomaly, "belief→anomaly"),
            (:mean_C, :mean_sigma, "confidence→sigma"),
        ]

        write(report, "### Granger F-tests\n\n")
        write(report, "| Pair | Lag | F | p | Sig |\n")
        write(report, "|------|-----|---|---|-----|\n")

        for (x_col, y_col, label) in pairs
            for lag in 1:min(3, div(nrow(ts), 5))
                result = _granger_f_test(Float64.(ts[!, x_col]), Float64.(ts[!, y_col]), lag)
                if result !== nothing
                    f_stat, p_val = result
                    sig = p_val < 0.01 ? "***" : p_val < 0.05 ? "**" : p_val < 0.1 ? "*" : ""
                    write(report, @sprintf("| %s | %d | %.3f | %.4f | %s |\n",
                        label, lag, f_stat, p_val, sig))
                    push!(stats_rows, ("S6", "granger_$(x_col)_$(y_col)_lag$(lag)",
                        f_stat, p_val, Float64(N), ""))
                end
            end
        end
        write(report, "\n")
    else
        write(report, "*Insufficient time series data.*\n\n")
    end

    # S7: CUSUM changepoint
    write(report, "## S7: CUSUM Changepoint Detection\n\n")
    if nrow(ts) >= 20
        for (col, label) in [(:mean_beff_A, "belief"), (:mean_sigma, "sigma"), (:mean_C, "confidence")]
            vals = Float64.(ts[!, col])
            mu = mean(vals)
            cusum = cumsum(vals .- mu)
            cp_idx = argmax(abs.(cusum))
            cp_tick = ts.tick[cp_idx]
            write(report, @sprintf("**%s:** changepoint at tick %d (CUSUM=%.3f)\n\n",
                label, cp_tick, cusum[cp_idx]))
            push!(stats_rows, ("S7", "cusum_cp_$(col)", Float64(cp_tick), NaN, cusum[cp_idx], ""))
        end
    else
        write(report, "*Insufficient data for CUSUM.*\n\n")
    end

    # ── Write outputs ──
    report_path = joinpath(OUTDIR, "causal_N$(N)_report.md")
    open(report_path, "w") do f
        write(f, String(take!(report)))
    end

    csv_path = joinpath(OUTDIR, "causal_N$(N)_summary.csv")
    csv_df = DataFrame(
        section = [r[1] for r in stats_rows],
        metric  = [r[2] for r in stats_rows],
        value   = [r[3] for r in stats_rows],
        p_value = [r[4] for r in stats_rows],
        extra   = [r[5] for r in stats_rows],
        note    = [r[6] for r in stats_rows],
    )
    CSV.write(csv_path, csv_df)

    progress_log("  → Report: $report_path")
    progress_log("  → Summary: $csv_path ($(nrow(csv_df)) statistics)")
end

# ══════════════════════════════════════════════════════════════
# Phase B: Cross-N Comparison Report
# ══════════════════════════════════════════════════════════════

function generate_comparison_report()
    progress_log("Generating cross-N comparison report...")

    io = IOBuffer()
    println(io, "# Large-N Scaling Comparison Report")
    println(io)
    println(io, "Generated: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    println(io)
    println(io, "---")
    println(io)

    # ═══ Section 1: Sweep convergence vs N ═══
    println(io, "## 1. Convergence Rate & Speed vs N (Sweep)")
    println(io)

    sweep_path = joinpath(OUTDIR, "sweep_N_extended.csv")
    if isfile(sweep_path)
        sweep_df = CSV.read(sweep_path, DataFrame)

        println(io, "### Convergence Rate by Condition")
        println(io)
        println(io, "| N | A_baseline | B_lockin | C_norm_only | D_full |")
        println(io, "|---|-----------|---------|------------|--------|")

        for N in N_ALL
            sub = filter(r -> r.N == N, sweep_df)
            if nrow(sub) == 0
                continue
            end
            parts = String[]
            for label in ["A_baseline", "B_lockin_only", "C_normative_only", "D_full_model"]
                cond_sub = filter(r -> r.condition == label, sub)
                if nrow(cond_sub) == 0
                    push!(parts, "—")
                else
                    rate = mean(cond_sub.converged)
                    push!(parts, @sprintf("%.0f%%", rate * 100))
                end
            end
            println(io, "| $N | $(join(parts, " | ")) |")
        end
        println(io)

        # Mean convergence tick
        println(io, "### Mean Convergence Tick (converged trials)")
        println(io)
        println(io, "| N | A_baseline | B_lockin | C_norm_only | D_full |")
        println(io, "|---|-----------|---------|------------|--------|")

        for N in N_ALL
            sub = filter(r -> r.N == N && r.converged, sweep_df)
            if nrow(sub) == 0
                continue
            end
            parts = String[]
            for label in ["A_baseline", "B_lockin_only", "C_normative_only", "D_full_model"]
                cond_sub = filter(r -> r.condition == label, sub)
                if nrow(cond_sub) == 0
                    push!(parts, "—")
                else
                    push!(parts, @sprintf("%.0f", mean(cond_sub.convergence_tick)))
                end
            end
            println(io, "| $N | $(join(parts, " | ")) |")
        end
        println(io)

        # Speed ratios (D/A)
        println(io, "### Speed Ratio D_full/A_baseline (behavioral layer)")
        println(io)
        println(io, "| N | D/A ratio | A mean tick | D mean tick |")
        println(io, "|---|-----------|------------|------------|")

        for N in N_ALL
            sub = filter(r -> r.N == N, sweep_df)
            a_ticks = filter(r -> r.condition == "A_baseline" && r.first_tick_behavioral > 0, sub).first_tick_behavioral
            d_ticks = filter(r -> r.condition == "D_full_model" && r.first_tick_behavioral > 0, sub).first_tick_behavioral
            if isempty(a_ticks) || isempty(d_ticks)
                println(io, "| $N | — | — | — |")
            else
                ma = mean(a_ticks)
                md = mean(d_ticks)
                println(io, @sprintf("| %d | %.1fx | %.0f | %.0f |", N, ma/md, ma, md))
            end
        end
        println(io)

        # Scaling exponent analysis
        println(io, "### Scaling Analysis: convergence_tick ∝ N^α")
        println(io)
        for label in ["A_baseline", "D_full_model"]
            conv = filter(r -> r.condition == label && r.converged, sweep_df)
            if nrow(conv) < 2
                continue
            end
            gdf = combine(groupby(conv, :N), :convergence_tick => mean => :mean_tick)
            sort!(gdf, :N)
            if nrow(gdf) >= 2
                log_N = log.(Float64.(gdf.N))
                log_t = log.(gdf.mean_tick)
                X = hcat(ones(length(log_N)), log_N)
                beta = X \ log_t
                alpha = beta[2]
                println(io, @sprintf("**%s:** α ≈ %.3f (log-log slope)", label, alpha))
                if alpha < 0.5
                    println(io, "  → **Sub-linear growth** — cascade time largely independent of N")
                elseif alpha < 1.0
                    println(io, "  → Sub-linear but non-trivial N dependence")
                else
                    println(io, "  → Linear or super-linear — N dependence detected")
                end
                println(io)
            end
        end
    else
        println(io, "*Sweep data not yet available.*")
        println(io)
    end

    # ═══ Section 2: Ablation effect sizes vs N ═══
    println(io, "## 2. Ablation Effect Sizes vs N")
    println(io)

    ablation_found = false
    println(io, "| N | Dyn_Norm conv% | Dyn_noNorm conv% | None_noNorm conv% | Norm speedup |")
    println(io, "|---|----------------|------------------|-------------------|-------------|")

    for N in N_ALL
        abl_path = joinpath(OUTDIR, "ablation_3x2_N$(N).csv")
        if !isfile(abl_path)
            continue
        end
        ablation_found = true
        adf = CSV.read(abl_path, DataFrame)

        dn = filter(r -> r.condition == "Dyn_Norm", adf)
        dnn = filter(r -> r.condition == "Dyn_noNorm", adf)
        nnn = filter(r -> r.condition == "None_noNorm", adf)

        dn_rate = nrow(dn) > 0 ? @sprintf("%.0f%%", mean(dn.converged) * 100) : "—"
        dnn_rate = nrow(dnn) > 0 ? @sprintf("%.0f%%", mean(dnn.converged) * 100) : "—"
        nnn_rate = nrow(nnn) > 0 ? @sprintf("%.0f%%", mean(nnn.converged) * 100) : "—"

        # Norm speedup: Dyn_noNorm / Dyn_Norm mean conv_tick
        dn_ticks = filter(r -> r.converged, dn).conv_tick
        dnn_ticks = filter(r -> r.converged, dnn).conv_tick
        if !isempty(dn_ticks) && !isempty(dnn_ticks)
            speedup = @sprintf("%.1fx", mean(dnn_ticks) / mean(dn_ticks))
        else
            speedup = "—"
        end

        println(io, "| $N | $dn_rate | $dnn_rate | $nnn_rate | $speedup |")
    end

    if !ablation_found
        println(io, "*Ablation data not yet available.*")
    end
    println(io)

    # ═══ Section 3: Cascade pathway metrics vs N ═══
    println(io, "## 3. Cascade Pathway Metrics vs N")
    println(io)

    println(io, "### A2 Pathway: θ=3.0, Φ=0.0")
    println(io)
    println(io, "| N | conv% | mean_diss | mean_enf | mean_churn | mean_flip | mean_tick |")
    println(io, "|---|-------|-----------|----------|------------|-----------|-----------|")

    for N in N_ALL
        cp = joinpath(OUTDIR, "cascade_pathways_N$(N).csv")
        if !isfile(cp)
            continue
        end
        cdf_pw = CSV.read(cp, DataFrame)
        sub = filter(r -> r.theta_crystal == 3.0 && r.Phi == 0.0, cdf_pw)
        if nrow(sub) == 0
            continue
        end
        n_conv = sum(sub.converged)
        conv_pct = @sprintf("%.0f%%", mean(sub.converged) * 100)
        md = @sprintf("%.1f", mean(sub.total_dissolutions))
        me = @sprintf("%.1f", mean(sub.total_enforcements))
        mc = @sprintf("%.1f", mean(sub.time_in_churn))
        mf = @sprintf("%.1f", mean(sub.recryst_flip))
        conv_sub = filter(r -> r.converged, sub)
        mt = isempty(conv_sub) ? "—" : @sprintf("%.0f", mean(conv_sub.convergence_tick))
        println(io, "| $N | $conv_pct | $md | $me | $mc | $mf | $mt |")
    end
    println(io)

    # ═══ Section 4: Granger F-statistics across N ═══
    println(io, "## 4. Granger Causality Stability Across N")
    println(io)
    println(io, "Key test: anomaly → sigma (dissolution proxy)")
    println(io)
    println(io, "| N | Lag1 F | Lag1 p | Lag2 F | Lag2 p |")
    println(io, "|---|--------|--------|--------|--------|")

    for N in N_ALL
        csv_path = joinpath(OUTDIR, "causal_N$(N)_summary.csv")
        if !isfile(csv_path)
            continue
        end
        sdf = CSV.read(csv_path, DataFrame)

        lag1 = filter(r -> r.metric == "granger_anomaly_mean_sigma_lag1", sdf)
        lag2 = filter(r -> r.metric == "granger_anomaly_mean_sigma_lag2", sdf)

        f1 = nrow(lag1) > 0 ? @sprintf("%.2f", lag1.value[1]) : "—"
        p1 = nrow(lag1) > 0 ? @sprintf("%.4f", lag1.p_value[1]) : "—"
        f2 = nrow(lag2) > 0 ? @sprintf("%.2f", lag2.value[1]) : "—"
        p2 = nrow(lag2) > 0 ? @sprintf("%.4f", lag2.p_value[1]) : "—"

        println(io, "| $N | $f1 | $p1 | $f2 | $p2 |")
    end
    println(io)

    # ═══ Section 5: CUSUM changepoints across N ═══
    println(io, "## 5. CUSUM Changepoint Timing Across N")
    println(io)
    println(io, "| N | Belief CP | Sigma CP | Confidence CP |")
    println(io, "|---|-----------|----------|---------------|")

    for N in N_ALL
        csv_path = joinpath(OUTDIR, "causal_N$(N)_summary.csv")
        if !isfile(csv_path)
            continue
        end
        sdf = CSV.read(csv_path, DataFrame)

        bel = filter(r -> r.metric == "cusum_cp_mean_beff_A", sdf)
        sig = filter(r -> r.metric == "cusum_cp_mean_sigma", sdf)
        con = filter(r -> r.metric == "cusum_cp_mean_C", sdf)

        b = nrow(bel) > 0 ? @sprintf("%.0f", bel.value[1]) : "—"
        s = nrow(sig) > 0 ? @sprintf("%.0f", sig.value[1]) : "—"
        c = nrow(con) > 0 ? @sprintf("%.0f", con.value[1]) : "—"

        println(io, "| $N | $b | $s | $c |")
    end
    println(io)

    # ═══ Section 6: Key findings ═══
    println(io, "## 6. Key Findings")
    println(io)
    println(io, "*(Auto-populated after all experiments complete)*")
    println(io)

    # Check if sweep data is available for conclusions
    if isfile(sweep_path)
        sweep_df = CSV.read(sweep_path, DataFrame)
        d_conv = filter(r -> r.condition == "D_full_model" && r.converged, sweep_df)
        if nrow(d_conv) > 0
            gdf = combine(groupby(d_conv, :N), :convergence_tick => mean => :mean_tick)
            sort!(gdf, :N)
            if nrow(gdf) >= 2
                log_N = log.(Float64.(gdf.N))
                log_t = log.(gdf.mean_tick)
                X = hcat(ones(length(log_N)), log_N)
                beta = X \ log_t
                alpha = beta[2]
                if alpha < 0.5
                    println(io, "1. **Cascade convergence time is sub-linear in N** (α≈$(round(alpha, digits=3)))")
                    println(io, "   - Supports the paper's claim that cascade timing is largely N-independent")
                else
                    println(io, "1. **Cascade convergence time shows meaningful N-dependence** (α≈$(round(alpha, digits=3)))")
                    println(io, "   - The paper's claim needs qualification for N > 500")
                end
                println(io)
            end
        end
    end

    # ── Write ──
    report_path = joinpath(OUTDIR, "comparison_report.md")
    open(report_path, "w") do f
        write(f, String(take!(io)))
    end
    progress_log("→ Comparison report: $report_path")
end

# ══════════════════════════════════════════════════════════════
# Main: CLI dispatch
# ══════════════════════════════════════════════════════════════

function main()
    mkpath(OUTDIR)

    args = ARGS
    if isempty(args)
        println("Usage:")
        println("  julia scripts/large_N_causal.jl N=1000     # Per-N analysis")
        println("  julia scripts/large_N_causal.jl all         # All N + comparison")
        println("  julia scripts/large_N_causal.jl report      # Comparison report only")
        return
    end

    arg = args[1]

    if arg == "report"
        generate_comparison_report()
    elseif arg == "all"
        for N in N_ALL
            pathway_path = joinpath(OUTDIR, "cascade_pathways_N$(N).csv")
            if isfile(pathway_path)
                run_per_N_causal(N)
            else
                progress_log("Skipping N=$N — no pathway data yet")
            end
        end
        generate_comparison_report()
    elseif startswith(arg, "N=")
        N = parse(Int, split(arg, "=")[2])
        run_per_N_causal(N)
    else
        println("Unknown argument: $arg")
        println("Use N=<value>, 'all', or 'report'")
    end

    progress_log("Done.")
end

main()
