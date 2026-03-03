#!/usr/bin/env julia
#
# Large-N 3×2 Ablation: Experiential Memory Level × Normative Memory
# Extension of run_ablation_3x2.jl to N ∈ {1000, 2000, 5000, 10000, 20000}
#
# Changes from original:
#   - N_TRIALS reduced from 100 to 30 (large-N variance is smaller)
#   - Adaptive T_MAX via select_T_MAX(N)
#   - Per-N output files: results/large_N/ablation_3x2_N{N}.csv
#   - Checkpoint/resume support
#
# Schema matches results/ablation_3x2_summary.csv exactly.
#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DualMemoryABM
using CSV
using Statistics
using Random

include(joinpath(@__DIR__, "large_N_common.jl"))

# ══════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════

const N_VALUES  = [1000, 2000, 5000, 10000, 20000]
const N_TRIALS  = 30
const BASE_SEED = 7000

const BEH_THRESH  = 0.95
const BEH_WINDOW  = 50

const OUTDIR = joinpath(@__DIR__, "..", "results", "large_N")

# ══════════════════════════════════════════════════════════════
# Condition definition (identical to run_ablation_3x2.jl)
# ══════════════════════════════════════════════════════════════

struct Condition
    label::String
    exp_level::Symbol       # :none, :fixed, :dynamic
    enable_normative::Bool
    w_base::Int
    w_max::Int
end

const CONDITIONS = [
    Condition("None_noNorm",  :none,    false, 5, 5),
    Condition("None_Norm",    :none,    true,  5, 5),
    Condition("Fixed_noNorm", :fixed,   false, 5, 5),
    Condition("Fixed_Norm",   :fixed,   true,  5, 5),
    Condition("Dyn_noNorm",   :dynamic, false, 2, 6),
    Condition("Dyn_Norm",     :dynamic, true,  2, 6),
]

# ══════════════════════════════════════════════════════════════
# Custom run loop with ablation (from run_ablation_3x2.jl)
# ══════════════════════════════════════════════════════════════

function run_ablation(N::Int, T_MAX::Int, cond::Condition, seed::Int)
    params = SimulationParams(
        N = N, T = T_MAX, seed = seed,
        w_base = cond.w_base, w_max = cond.w_max,
        enable_normative = cond.enable_normative,
        V = 0, Phi = 0.0,
    )
    agents, ws, history, rng = initialize(params)
    probes = ProbeSet()
    init_probes!(probes, params, rng)
    tick_count = 0

    beh_counter = 0
    beh_converged = false
    beh_conv_tick = 0

    for t in 1:T_MAX
        run_tick!(t, agents, ws, history, tick_count, params, rng, probes)
        tick_count += 1

        # ── Ablation: freeze experiential learning ──
        if cond.exp_level == :none
            for i in 1:N
                agents[i].b_exp_A = 0.5
            end
        end

        # ── Behavioral convergence check ──
        m = history[tick_count]
        majority = max(m.fraction_A, 1.0 - m.fraction_A)
        if majority >= BEH_THRESH
            beh_counter += 1
            if !beh_converged && beh_counter >= BEH_WINDOW
                beh_converged = true
                beh_conv_tick = t - BEH_WINDOW + 1
            end
        else
            beh_counter = 0
        end

        if beh_converged && t >= beh_conv_tick + 200
            break
        end
    end

    return (
        history    = history[1:tick_count],
        params     = params,
        converged  = beh_converged,
        conv_tick  = beh_conv_tick,
        tick_count = tick_count,
    )
end

# ══════════════════════════════════════════════════════════════
# Post-hoc metrics (from run_ablation_3x2.jl)
# ══════════════════════════════════════════════════════════════

function compute_metrics(result, N::Int)
    hist = result.history
    n = length(hist)
    final = hist[end]

    first_beh_tick = 0
    for m in hist
        if max(m.fraction_A, 1.0 - m.fraction_A) >= BEH_THRESH
            first_beh_tick = m.tick
            break
        end
    end

    first_crystal_50 = 0
    for m in hist
        if m.num_crystallised >= N ÷ 2
            first_crystal_50 = m.tick
            break
        end
    end

    mean_enforce = 0.0
    if result.converged
        ct = result.conv_tick
        start_idx = 0
        for idx in 1:n
            if hist[idx].tick >= ct
                start_idx = idx
                break
            end
        end
        if start_idx > 0
            end_idx = min(start_idx + 99, n)
            mean_enforce = mean(hist[idx].num_enforcements for idx in start_idx:end_idx)
        end
    end

    return (
        converged         = result.converged,
        conv_tick         = result.conv_tick,
        first_beh_tick    = first_beh_tick,
        first_crystal_50  = first_crystal_50,
        final_coord       = final.coordination_rate,
        final_majority    = max(final.fraction_A, 1.0 - final.fraction_A),
        final_cryst       = final.num_crystallised,
        final_norm_str    = final.mean_norm_strength,
        final_belief_err  = final.belief_error,
        mean_enforce      = mean_enforce,
        total_ticks       = result.tick_count,
    )
end

# ══════════════════════════════════════════════════════════════
# Run sweep
# ══════════════════════════════════════════════════════════════

function run_sweep()
    mkpath(OUTDIR)

    for N in N_VALUES
        T_MAX = select_T_MAX(N)
        outpath = joinpath(OUTDIR, "ablation_3x2_N$(N).csv")

        # Checkpoint
        done = load_completed_trials(outpath, [:condition, :trial])
        if length(done) >= length(CONDITIONS) * N_TRIALS
            progress_log("N=$N — all $(length(CONDITIONS) * N_TRIALS) trials done, skipping")
            continue
        end
        if !isempty(done)
            progress_log("N=$N — resuming ($(length(done)) trials already completed)")
        end

        progress_log("N=$N  T_MAX=$T_MAX  ($N_TRIALS trials × $(length(CONDITIONS)) conditions)")
        t_start = time()

        for cond in CONDITIONS
            needed = [t for t in 1:N_TRIALS if (cond.label, t) ∉ done]
            if isempty(needed)
                progress_log("  $(cond.label) — all done, skipping")
                continue
            end

            progress_log("  $(cond.label) × $(length(needed)) trials...")

            block_rows = Vector{Any}()
            lk = ReentrantLock()

            Threads.@threads for trial in needed
                seed = hash((BASE_SEED, N, cond.label, trial)) % typemax(Int)
                result = run_ablation(N, T_MAX, cond, Int(seed))
                met = compute_metrics(result, N)

                row = (
                    N              = N,
                    condition      = cond.label,
                    exp_level      = string(cond.exp_level),
                    norm_on        = cond.enable_normative,
                    trial          = trial,
                    converged      = met.converged,
                    conv_tick      = met.conv_tick,
                    first_beh_tick = met.first_beh_tick,
                    first_cryst_50 = met.first_crystal_50,
                    final_coord    = met.final_coord,
                    final_majority = met.final_majority,
                    final_cryst    = met.final_cryst,
                    final_norm_str = met.final_norm_str,
                    final_belief_err = met.final_belief_err,
                    mean_enforce   = met.mean_enforce,
                    total_ticks    = met.total_ticks,
                )

                lock(lk) do
                    push!(block_rows, row)
                end
            end

            incremental_csv_write(outpath, block_rows)
        end

        elapsed = round(time() - t_start, digits=1)
        progress_log("  N=$N complete in $(elapsed)s")
    end
end

# ══════════════════════════════════════════════════════════════
# Console summary
# ══════════════════════════════════════════════════════════════

function print_summary()
    exp_levels = ["none", "fixed", "dynamic"]
    exp_labels = ["None", "Fixed(w=5)", "Dynamic[2,6]"]

    for N in N_VALUES
        outpath = joinpath(OUTDIR, "ablation_3x2_N$(N).csv")
        if !isfile(outpath)
            continue
        end
        df = CSV.read(outpath, DataFrame)
        rows = [NamedTuple(row) for row in eachrow(df)]

        println("\n" * "=" ^ 90)
        println("N = $N   ($N_TRIALS trials per condition)")
        println("=" ^ 90)

        println("\n  ── 3×2 Grid: Experiential Level × Normative ──\n")
        println("  ", rpad("", 18), rpad("Norm OFF", 28), rpad("Norm ON", 28))
        println("  ", "-" ^ 74)

        for (i, exp_lv) in enumerate(exp_levels)
            for norm_on in [false, true]
                sub = [r for r in rows if r.exp_level == exp_lv && r.norm_on == norm_on]
                ns = length(sub)
                n_conv = count(r -> r.converged, sub)
                conv_ticks = [r.conv_tick for r in sub if r.converged]
                mean_spd = length(conv_ticks) > 0 ? round(mean(conv_ticks), digits=1) : NaN
                if norm_on == false
                    print("  ", rpad(exp_labels[i], 18))
                    print(rpad("$(n_conv)/$(ns) / $(isnan(mean_spd) ? "—" : string(mean_spd))", 28))
                else
                    println(rpad("$(n_conv)/$(ns) / $(isnan(mean_spd) ? "—" : string(mean_spd))", 28))
                end
            end
        end

        # Speedup from Norm
        println("\n  ── Norm ON speedup (mean tick: OFF → ON) ──")
        for (i, exp_lv) in enumerate(exp_levels)
            sub_off = [r for r in rows if r.exp_level == exp_lv && r.norm_on == false]
            sub_on  = [r for r in rows if r.exp_level == exp_lv && r.norm_on == true]
            ct_off = [r.conv_tick for r in sub_off if r.converged]
            ct_on  = [r.conv_tick for r in sub_on  if r.converged]
            off_str = isempty(ct_off) ? "—" : string(round(mean(ct_off), digits=1))
            on_str  = isempty(ct_on)  ? "—" : string(round(mean(ct_on), digits=1))
            if off_str != "—" && on_str != "—"
                ratio = round(parse(Float64, off_str) / parse(Float64, on_str), digits=1)
                println("  ", rpad(exp_labels[i], 18), "$off_str → $on_str  ($(ratio)× speedup)")
            else
                println("  ", rpad(exp_labels[i], 18), "$off_str → $on_str")
            end
        end
    end
end

# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

function main()
    progress_log("Large-N 3×2 Ablation: N ∈ $(N_VALUES)")
    t0 = time()
    run_sweep()
    print_summary()
    elapsed = round(time() - t0, digits=1)
    progress_log("Total time: $(elapsed)s")
    println("\nDone.")
end

main()
