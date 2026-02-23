#!/usr/bin/env julia
#
# C2: Perturbation Robustness — DualMemory vs EWA
#
# Protocol (per comparison_experiment_spec.md):
#   1. Run simulation for 500 ticks (no early termination)
#   2. At tick 500, randomly reset SHOCK_FRAC of agents to naive state
#   3. Continue for 500 more ticks (total T=1000)
#
# Perturbation resets:
#   EWA: attract_A = attract_B = 0, prob_A = 0.5, N_exp = 1.0
#   DM:  FIFO cleared, b_exp_A = 0.5, C = C0, w = w_init,
#        r = NO_NORM, sigma = 0, e = 0, a = 0
#
# Output:
#   results/c2_trajectories.csv — tick-by-tick for seed=42, each (model, shock_frac)
#   results/c2_summary.csv     — per-trial recovery metrics
#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DualMemoryABM
using CSV
using Statistics
using Random

# ══════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════

const BEST_LAMBDA     = 10.0
const N_AGENTS        = 100
const T_PRE           = 500       # ticks before perturbation
const T_POST          = 500       # ticks after perturbation
const T_TOTAL         = T_PRE + T_POST
const SHOCK_FRACS     = [0.2, 0.5]   # 20% (standard) and 50% (extreme)
const N_TRIALS        = 50
const BASE_SEED       = 7000
const RECOVERY_THRESH = 0.90

# ══════════════════════════════════════════════════════════════
# Perturbation Functions
# ══════════════════════════════════════════════════════════════

"""
    perturb_ewa!(agents, shocked_ids)

Reset selected EWA agents to naive state: zero attractions, p=0.5.
"""
function perturb_ewa!(agents::Vector{EWAAgent}, shocked_ids::Vector{Int})
    for i in shocked_ids
        agents[i].attract_A = 0.0
        agents[i].attract_B = 0.0
        agents[i].N_exp = 1.0
        agents[i].prob_A = 0.5
    end
end

"""
    perturb_dm!(agents, shocked_ids, params)

Reset selected DM agents to naive state: clear FIFO, reset confidence
and normative memory. These agents become "newcomers" with no memory.
"""
function perturb_dm!(agents::Vector{AgentState}, shocked_ids::Vector{Int},
                     params::SimulationParams)
    w_init = params.w_base + floor(Int, params.C0 * (params.w_max - params.w_base))
    for i in shocked_ids
        # Clear experiential memory
        agents[i].fifo = RingBuffer(params.w_max)
        agents[i].b_exp_A = 0.5
        # Reset confidence
        agents[i].C = params.C0
        agents[i].w = w_init
        # Dissolve normative memory
        agents[i].r = NO_NORM
        agents[i].sigma = 0.0
        agents[i].e = 0.0
        agents[i].a = 0
        # Clear enforcement buffer and derived fields
        agents[i].pending_signal = NO_SIGNAL
        agents[i].compliance = 0.0
        agents[i].b_eff_A = 0.5
    end
end

# ══════════════════════════════════════════════════════════════
# Custom Run Loops (no early termination)
# ══════════════════════════════════════════════════════════════

function run_ewa_c2(seed::Int, shock_frac::Float64)
    params = EWAParams(
        N=N_AGENTS, T=T_TOTAL, seed=seed,
        delta=1.0, phi=0.9, rho=0.9, lambda=BEST_LAMBDA,
    )
    agents, ws, history, rng = ewa_initialize(params)
    tick_count = 0

    # Phase 1: pre-perturbation
    for t in 1:T_PRE
        ewa_run_tick!(t, agents, ws, history, tick_count, params, rng)
        tick_count += 1
    end

    pre_shock_coord = history[tick_count].coordination_rate

    # Shock
    n_shock = round(Int, shock_frac * N_AGENTS)
    shocked_ids = sort(shuffle(rng, collect(1:N_AGENTS))[1:n_shock])
    perturb_ewa!(agents, shocked_ids)

    # Phase 2: post-perturbation
    for t in (T_PRE + 1):T_TOTAL
        ewa_run_tick!(t, agents, ws, history, tick_count, params, rng)
        tick_count += 1
    end

    return history[1:tick_count], pre_shock_coord
end

function run_dm_c2(seed::Int, shock_frac::Float64)
    params = SimulationParams(
        N=N_AGENTS, T=T_TOTAL, seed=seed,
        w_base=2, w_max=6,
        enable_normative=true, V=3, Phi=1.0,
    )
    agents, ws, history, rng = initialize(params)
    probes = ProbeSet()
    init_probes!(probes, params, rng)
    tick_count = 0

    # Phase 1: pre-perturbation
    for t in 1:T_PRE
        run_tick!(t, agents, ws, history, tick_count, params, rng, probes)
        tick_count += 1
    end

    pre_shock_coord = history[tick_count].coordination_rate
    pre_shock_cryst = history[tick_count].num_crystallised

    # Shock
    n_shock = round(Int, shock_frac * N_AGENTS)
    shocked_ids = sort(shuffle(rng, collect(1:N_AGENTS))[1:n_shock])
    perturb_dm!(agents, shocked_ids, params)

    # Phase 2: post-perturbation
    for t in (T_PRE + 1):T_TOTAL
        run_tick!(t, agents, ws, history, tick_count, params, rng, probes)
        tick_count += 1
    end

    return history[1:tick_count], pre_shock_coord, pre_shock_cryst
end

# ══════════════════════════════════════════════════════════════
# Metrics Computation
# ══════════════════════════════════════════════════════════════

function compute_c2_metrics(history, pre_shock_coord)
    # Post-shock coordination (first tick after shock)
    post_shock_coord = history[T_PRE + 1].coordination_rate
    coord_drop = post_shock_coord - pre_shock_coord

    # Recovery speed: first tick after T_PRE where coord >= threshold
    recovery_tick = 0
    for idx in (T_PRE + 1):length(history)
        if history[idx].coordination_rate >= RECOVERY_THRESH
            recovery_tick = history[idx].tick
            break
        end
    end
    recovered = recovery_tick > 0
    recovery_ticks = recovered ? recovery_tick - T_PRE : 0

    # Min coordination during post-shock phase
    min_coord = minimum(history[idx].coordination_rate for idx in (T_PRE + 1):length(history))

    # Steady-state coordination: mean over last 100 ticks (ticks 901-1000)
    ss_start = length(history) - 99
    ss_end = length(history)
    ss_coord = mean(history[idx].coordination_rate for idx in ss_start:ss_end)

    # Post-shock crystallisation (DM only — 0 for EWA)
    post_shock_cryst = history[T_PRE + 1].num_crystallised
    final_cryst = history[end].num_crystallised

    return (
        pre_shock_coord  = pre_shock_coord,
        post_shock_coord = post_shock_coord,
        coord_drop       = coord_drop,
        min_coord        = min_coord,
        recovery_tick    = recovery_tick,
        recovered        = recovered,
        recovery_ticks   = recovery_ticks,
        ss_coord         = ss_coord,
        post_shock_cryst = post_shock_cryst,
        final_cryst      = final_cryst,
    )
end

# ══════════════════════════════════════════════════════════════
# Part 1: Representative Trajectories
# ══════════════════════════════════════════════════════════════

function write_trajectories()
    println("=" ^ 60)
    println("Part 1: Representative trajectories (seed=42)")
    println("=" ^ 60)

    traj_rows = []

    for shock_frac in SHOCK_FRACS
        shock_pct = round(Int, shock_frac * 100)

        # EWA_BL
        label = "EWA_BL_$(shock_pct)pct"
        println("  Running $label ...")
        hist, _ = run_ewa_c2(42, shock_frac)
        for m in hist
            push!(traj_rows, (
                condition          = label,
                shock_frac         = shock_frac,
                tick               = m.tick,
                fraction_A         = m.fraction_A,
                coordination_rate  = m.coordination_rate,
                mean_confidence    = m.mean_confidence,
                num_crystallised   = m.num_crystallised,
                mean_norm_strength = m.mean_norm_strength,
                belief_error       = m.belief_error,
                num_enforcements   = m.num_enforcements,
            ))
        end

        # DM_full
        label = "DM_full_$(shock_pct)pct"
        println("  Running $label ...")
        hist, _, _ = run_dm_c2(42, shock_frac)
        for m in hist
            push!(traj_rows, (
                condition          = label,
                shock_frac         = shock_frac,
                tick               = m.tick,
                fraction_A         = m.fraction_A,
                coordination_rate  = m.coordination_rate,
                mean_confidence    = m.mean_confidence,
                num_crystallised   = m.num_crystallised,
                mean_norm_strength = m.mean_norm_strength,
                belief_error       = m.belief_error,
                num_enforcements   = m.num_enforcements,
            ))
        end
    end

    outpath = joinpath(@__DIR__, "..", "results", "c2_trajectories.csv")
    CSV.write(outpath, traj_rows)
    println("  Saved $(length(traj_rows)) trajectory rows → $outpath")
end

# ══════════════════════════════════════════════════════════════
# Part 2: Multi-Trial Summary
# ══════════════════════════════════════════════════════════════

function write_summary()
    println("\n" * "=" ^ 60)
    println("Part 2: Multi-trial summary ($N_TRIALS trials per condition)")
    println("=" ^ 60)

    summary_rows = []
    lk = ReentrantLock()

    for shock_frac in SHOCK_FRACS
        shock_pct = round(Int, shock_frac * 100)

        # EWA_BL
        label = "EWA_BL"
        println("  Running $label @ $(shock_pct)% shock × $N_TRIALS trials...")
        Threads.@threads for trial in 1:N_TRIALS
            seed = hash((BASE_SEED, label, shock_frac, trial)) % typemax(Int)
            hist, pre_coord = run_ewa_c2(Int(seed), shock_frac)
            met = compute_c2_metrics(hist, pre_coord)
            lock(lk) do
                push!(summary_rows, (
                    condition        = label,
                    shock_frac       = shock_frac,
                    trial            = trial,
                    pre_shock_coord  = met.pre_shock_coord,
                    post_shock_coord = met.post_shock_coord,
                    coord_drop       = met.coord_drop,
                    min_coord        = met.min_coord,
                    recovery_tick    = met.recovery_tick,
                    recovered        = met.recovered,
                    recovery_ticks   = met.recovery_ticks,
                    ss_coord         = met.ss_coord,
                    post_shock_cryst = met.post_shock_cryst,
                    final_cryst      = met.final_cryst,
                ))
            end
        end

        # DM_full
        label = "DM_full"
        println("  Running $label @ $(shock_pct)% shock × $N_TRIALS trials...")
        Threads.@threads for trial in 1:N_TRIALS
            seed = hash((BASE_SEED, label, shock_frac, trial)) % typemax(Int)
            hist, pre_coord, _ = run_dm_c2(Int(seed), shock_frac)
            met = compute_c2_metrics(hist, pre_coord)
            lock(lk) do
                push!(summary_rows, (
                    condition        = label,
                    shock_frac       = shock_frac,
                    trial            = trial,
                    pre_shock_coord  = met.pre_shock_coord,
                    post_shock_coord = met.post_shock_coord,
                    coord_drop       = met.coord_drop,
                    min_coord        = met.min_coord,
                    recovery_tick    = met.recovery_tick,
                    recovered        = met.recovered,
                    recovery_ticks   = met.recovery_ticks,
                    ss_coord         = met.ss_coord,
                    post_shock_cryst = met.post_shock_cryst,
                    final_cryst      = met.final_cryst,
                ))
            end
        end
    end

    outpath = joinpath(@__DIR__, "..", "results", "c2_summary.csv")
    CSV.write(outpath, summary_rows)
    println("  Saved $(length(summary_rows)) summary rows → $outpath")

    return summary_rows
end

# ══════════════════════════════════════════════════════════════
# Part 3: Console Summary Table
# ══════════════════════════════════════════════════════════════

function print_summary(summary_rows)
    println("\n" * "=" ^ 90)
    println("C2 PERTURBATION ROBUSTNESS — SUMMARY")
    println("=" ^ 90)

    for shock_frac in SHOCK_FRACS
        shock_pct = round(Int, shock_frac * 100)
        println("\n── Shock = $(shock_pct)% ─────────────────────────────────────────")
        println(rpad("Metric", 28), rpad("EWA_BL", 16), rpad("DM_full", 16), "  Δ")
        println("-" ^ 76)

        ewa = [r for r in summary_rows if r.condition == "EWA_BL" && r.shock_frac == shock_frac]
        dm  = [r for r in summary_rows if r.condition == "DM_full" && r.shock_frac == shock_frac]

        # Recovery rate
        ewa_rec = count(r -> r.recovered, ewa)
        dm_rec  = count(r -> r.recovered, dm)
        println(rpad("Recovery rate", 28),
                rpad("$ewa_rec/$(length(ewa))", 16),
                rpad("$dm_rec/$(length(dm))", 16))

        # Mean coord drop
        ewa_drop = round(mean([r.coord_drop for r in ewa]), digits=4)
        dm_drop  = round(mean([r.coord_drop for r in dm]), digits=4)
        println(rpad("Mean coord drop", 28),
                rpad(string(ewa_drop), 16),
                rpad(string(dm_drop), 16))

        # Mean min coordination
        ewa_min = round(mean([r.min_coord for r in ewa]), digits=4)
        dm_min  = round(mean([r.min_coord for r in dm]), digits=4)
        println(rpad("Mean min coordination", 28),
                rpad(string(ewa_min), 16),
                rpad(string(dm_min), 16))

        # Mean recovery ticks (among recovered)
        ewa_rticks = [r.recovery_ticks for r in ewa if r.recovered]
        dm_rticks  = [r.recovery_ticks for r in dm  if r.recovered]
        ewa_mean_r = length(ewa_rticks) > 0 ? round(mean(ewa_rticks), digits=1) : NaN
        dm_mean_r  = length(dm_rticks) > 0 ? round(mean(dm_rticks), digits=1) : NaN
        delta_r = (!isnan(ewa_mean_r) && !isnan(dm_mean_r)) ?
                  round(ewa_mean_r - dm_mean_r, digits=1) : NaN
        println(rpad("Mean recovery ticks", 28),
                rpad(isnan(ewa_mean_r) ? "N/A" : string(ewa_mean_r), 16),
                rpad(isnan(dm_mean_r) ? "N/A" : string(dm_mean_r), 16),
                isnan(delta_r) ? "" : "  EWA slower by $delta_r")

        # Median recovery ticks
        ewa_med_r = length(ewa_rticks) > 0 ? round(median(ewa_rticks), digits=1) : NaN
        dm_med_r  = length(dm_rticks) > 0 ? round(median(dm_rticks), digits=1) : NaN
        println(rpad("Median recovery ticks", 28),
                rpad(isnan(ewa_med_r) ? "N/A" : string(ewa_med_r), 16),
                rpad(isnan(dm_med_r) ? "N/A" : string(dm_med_r), 16))

        # Steady-state coordination (last 100 ticks)
        ewa_ss = round(mean([r.ss_coord for r in ewa]), digits=4)
        dm_ss  = round(mean([r.ss_coord for r in dm]), digits=4)
        println(rpad("Mean SS coordination", 28),
                rpad(string(ewa_ss), 16),
                rpad(string(dm_ss), 16))

        # DM-specific: crystallisation
        dm_post_cryst = round(mean([r.post_shock_cryst for r in dm]), digits=1)
        dm_final_cryst = round(mean([r.final_cryst for r in dm]), digits=1)
        println(rpad("DM post-shock crystallised", 28),
                rpad("—", 16),
                rpad(string(dm_post_cryst), 16))
        println(rpad("DM final crystallised", 28),
                rpad("—", 16),
                rpad(string(dm_final_cryst), 16))
    end
end

# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

function main()
    mkpath(joinpath(@__DIR__, "..", "results"))

    write_trajectories()
    summary_rows = write_summary()
    print_summary(summary_rows)

    println("\n\nDone.")
end

main()
