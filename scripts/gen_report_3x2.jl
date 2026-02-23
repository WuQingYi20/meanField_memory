#!/usr/bin/env julia
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CSV, Statistics, DataFrames

df = CSV.read(joinpath(@__DIR__, "..", "results", "ablation_3x2_summary.csv"), DataFrame)

exp_order = ["none", "fixed", "dynamic"]
exp_lab = Dict("none"=>"None (frozen)", "fixed"=>"Fixed (w=5)", "dynamic"=>"Dynamic [2,6]")

io = IOBuffer()

println(io, "# 3x2 Ablation Experiment: Experiential Memory Level x Normative Memory")
println(io)
println(io, "**Date**: 2026-02-20  ")
println(io, "**Script**: `scripts/run_ablation_3x2.jl`  ")
println(io, "**Raw data**: `results/ablation_3x2_summary.csv`  ")
println(io, "**Trials**: 100 per cell  ")
println(io)

println(io, "## Experimental Design")
println(io)
println(io, "3x2 factorial crossed with 3 population sizes (N = 20, 100, 500).")
println(io)
println(io, "**Factor 1 -- Experiential Memory (3 levels):**")
println(io, "| Level | Description | Parameters |")
println(io, "|-------|-------------|------------|")
println(io, "| None | `b_exp_A` reset to 0.5 after every tick -- no individual learning | -- |")
println(io, "| Fixed | Standard FIFO learning, fixed window | w_base=5, w_max=5 |")
println(io, "| Dynamic | Standard FIFO learning, confidence-driven window | w_base=2, w_max=6 |")
println(io)
println(io, "**Factor 2 -- Normative Memory:** OFF / ON (`enable_normative`)")
println(io)
println(io, "**Other parameters:** V=0, Phi=0.0, T=3000, alpha=0.1, beta=0.3, C0=0.5")
println(io)
println(io, "**Convergence criterion:** Behavioral majority >= 0.95 sustained for 50 consecutive ticks")
println(io)

# ────── Per-N tables ──────
for N in [20, 100, 500]
    sub = filter(r -> r.N == N, df)

    println(io, "---")
    println(io, "## N = $N")
    println(io)

    # ── Table 1: Convergence ──
    println(io, "### Convergence Rate and Speed")
    println(io)
    println(io, "| Exp Level | Norm OFF conv | Norm OFF mean tick | Norm ON conv | Norm ON mean tick | Speedup |")
    println(io, "|-----------|:---:|---:|:---:|---:|---:|")

    for exp_lv in exp_order
        s_off = filter(r -> r.exp_level == exp_lv && r.norm_on == false, sub)
        s_on  = filter(r -> r.exp_level == exp_lv && r.norm_on == true, sub)
        nt = nrow(s_off)

        n_off = count(r -> r.converged, eachrow(s_off))
        n_on  = count(r -> r.converged, eachrow(s_on))

        ct_off = [r.conv_tick for r in eachrow(s_off) if r.converged]
        ct_on  = [r.conv_tick for r in eachrow(s_on) if r.converged]

        m_off = length(ct_off) > 0 ? round(mean(ct_off), digits=1) : NaN
        m_on  = length(ct_on) > 0 ? round(mean(ct_on), digits=1) : NaN

        m_off_s = isnan(m_off) ? "--" : string(m_off)
        m_on_s  = isnan(m_on) ? "--" : string(m_on)

        if !isnan(m_off) && !isnan(m_on)
            spd = string(round(m_off / m_on, digits=1)) * "x"
        else
            spd = "--"
        end

        println(io, "| $(exp_lab[exp_lv]) | $n_off/$nt | $m_off_s | $n_on/$nt | $m_on_s | $spd |")
    end
    println(io)

    # ── Table 2: Final majority ──
    println(io, "### Final Majority Fraction (all 100 trials, including non-converged)")
    println(io)
    println(io, "| Exp Level | | Mean | Median | Min | Max | Std |")
    println(io, "|-----------|------|---:|---:|---:|---:|---:|")

    for exp_lv in exp_order
        for norm_on in [false, true]
            s = filter(r -> r.exp_level == exp_lv && r.norm_on == norm_on, sub)
            vals = s.final_majority

            v_mean = round(mean(vals), digits=3)
            v_med  = round(median(vals), digits=3)
            v_min  = round(minimum(vals), digits=3)
            v_max  = round(maximum(vals), digits=3)
            v_std  = round(std(vals), digits=3)

            norm_str = norm_on ? "Norm ON" : "Norm OFF"
            label = norm_on ? "" : exp_lab[exp_lv]

            println(io, "| $label | $norm_str | $v_mean | $v_med | $v_min | $v_max | $v_std |")
        end
    end
    println(io)

    # ── Table 3: Final coordination ──
    println(io, "### Final Coordination Rate (all 100 trials)")
    println(io)
    println(io, "| Exp Level | | Mean | Median | Min | Max |")
    println(io, "|-----------|------|---:|---:|---:|---:|")

    for exp_lv in exp_order
        for norm_on in [false, true]
            s = filter(r -> r.exp_level == exp_lv && r.norm_on == norm_on, sub)
            vals = s.final_coord

            v_mean = round(mean(vals), digits=3)
            v_med  = round(median(vals), digits=3)
            v_min  = round(minimum(vals), digits=2)
            v_max  = round(maximum(vals), digits=2)

            norm_str = norm_on ? "Norm ON" : "Norm OFF"
            label = norm_on ? "" : exp_lab[exp_lv]

            println(io, "| $label | $norm_str | $v_mean | $v_med | $v_min | $v_max |")
        end
    end
    println(io)
end

# ────── Cross-N summary ──────
println(io, "---")
println(io, "## Cross-N Summary")
println(io)

println(io, "### Convergence rate across N")
println(io)
println(io, "| Condition | N=20 | N=100 | N=500 |")
println(io, "|-----------|:---:|:---:|:---:|")

for exp_lv in exp_order
    for norm_on in [false, true]
        norm_str = norm_on ? "+Norm" : ""
        label = exp_lab[exp_lv] * norm_str
        vals = String[]
        for N in [20, 100, 500]
            s = filter(r -> r.N == N && r.exp_level == exp_lv && r.norm_on == norm_on, df)
            n_conv = count(r -> r.converged, eachrow(s))
            push!(vals, "$n_conv/$(nrow(s))")
        end
        println(io, "| $label | $(vals[1]) | $(vals[2]) | $(vals[3]) |")
    end
end
println(io)

println(io, "### Mean convergence tick across N (converged trials only)")
println(io)
println(io, "| Condition | N=20 | N=100 | N=500 |")
println(io, "|-----------|---:|---:|---:|")

for exp_lv in exp_order
    for norm_on in [false, true]
        norm_str = norm_on ? "+Norm" : ""
        label = exp_lab[exp_lv] * norm_str
        vals = String[]
        for N in [20, 100, 500]
            s = filter(r -> r.N == N && r.exp_level == exp_lv && r.norm_on == norm_on, df)
            ct = [r.conv_tick for r in eachrow(s) if r.converged]
            m = length(ct) > 0 ? string(round(mean(ct), digits=1)) : "--"
            push!(vals, m)
        end
        println(io, "| $label | $(vals[1]) | $(vals[2]) | $(vals[3]) |")
    end
end
println(io)

println(io, "### Mean final majority across N (all trials)")
println(io)
println(io, "| Condition | N=20 | N=100 | N=500 |")
println(io, "|-----------|---:|---:|---:|")

for exp_lv in exp_order
    for norm_on in [false, true]
        norm_str = norm_on ? "+Norm" : ""
        label = exp_lab[exp_lv] * norm_str
        vals = String[]
        for N in [20, 100, 500]
            s = filter(r -> r.N == N && r.exp_level == exp_lv && r.norm_on == norm_on, df)
            m = round(mean(s.final_majority), digits=3)
            push!(vals, string(m))
        end
        println(io, "| $label | $(vals[1]) | $(vals[2]) | $(vals[3]) |")
    end
end
println(io)

# ────── Norm speedup table ──────
println(io, "### Normative speedup ratio across N")
println(io)
println(io, "| Exp Level | N=20 | N=100 | N=500 |")
println(io, "|-----------|---:|---:|---:|")

for exp_lv in ["fixed", "dynamic"]
    vals = String[]
    for N in [20, 100, 500]
        s_off = filter(r -> r.N == N && r.exp_level == exp_lv && r.norm_on == false, df)
        s_on  = filter(r -> r.N == N && r.exp_level == exp_lv && r.norm_on == true, df)
        ct_off = [r.conv_tick for r in eachrow(s_off) if r.converged]
        ct_on  = [r.conv_tick for r in eachrow(s_on) if r.converged]
        if length(ct_off) > 0 && length(ct_on) > 0
            ratio = round(mean(ct_off) / mean(ct_on), digits=1)
            push!(vals, string(ratio) * "x")
        else
            push!(vals, "--")
        end
    end
    println(io, "| $(exp_lab[exp_lv]) | $(vals[1]) | $(vals[2]) | $(vals[3]) |")
end
println(io)

# ────── Key findings ──────
println(io, "---")
println(io, "## Key Findings")
println(io)
println(io, "### 1. Experiential memory is necessary for convergence")
println(io, "Without experiential learning (None row), convergence rate is 0/100 at all N, regardless of normative memory. With normative memory on, the final majority reaches ~0.62 (vs ~0.53 without) -- a modest improvement from crystallization, but far below the 0.95 convergence threshold. **Norms can amplify existing patterns but cannot create them from scratch.**")
println(io)
println(io, "### 2. Normative memory dramatically accelerates convergence")
println(io, "When experiential learning is present, adding normative memory reduces convergence time by 3-39x depending on conditions. The speedup is largest when experiential learning is weakest (Fixed at large N): at N=500, Fixed goes from 24/100 converged at 2079 ticks to 100/100 at 54 ticks (38.7x speedup).")
println(io)
println(io, "### 3. Normative memory compensates for weak experiential learning")
println(io, "Fixed(w=5)+Norm and Dynamic[2,6]+Norm converge at nearly identical speeds (within ~10 ticks across all N). The normative cascade mechanism makes the quality of experiential learning largely irrelevant once it provides a sufficient symmetry-breaking signal.")
println(io)
println(io, "### 4. Normative memory provides population-size robustness")
println(io, "Without norms, convergence time scales roughly linearly with N (and Fixed even loses convergence reliability at large N: 24% at N=500). With norms, convergence time grows very slowly with N (~25t to ~43t for Dynamic+Norm from N=20 to N=500) and convergence rate stays at 100%.")
println(io)
println(io, "### 5. Complementary roles")
println(io, "- **Experiential memory** = symmetry breaker: individual learning from partner actions creates a population-level behavioral majority")
println(io, "- **Normative memory** = pattern amplifier: DDM crystallization cascade detects the emerging majority, locks it in as an institutional norm, and drives the population to complete consensus")
println(io, "- Neither system alone achieves reliable, scalable convergence. Together they produce fast, complete, and robust norm emergence.")

outpath = joinpath(@__DIR__, "..", "results", "ablation_3x2_report.md")
open(outpath, "w") do f
    write(f, String(take!(io)))
end

println("Report written to $outpath")
