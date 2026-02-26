#!/usr/bin/env julia
#
# Network Topology Sweep
# Tests how ring, small-world, and scale-free topologies affect convergence
# compared to complete-graph (mean-field) baseline.
#
# Conditions:
#   Topologies: complete, ring, smallworld (p∈{0.01,0.1,0.5}), scalefree
#   Degrees: k ∈ {4, 8, 16}; BA uses m = k÷2
#   Population: N ∈ {100, 500}
#   Memory: Dynamic + Norm (best from 3×2 ablation), V=0, Φ=0
#   Trials: 50 per cell, T_max=5000
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

const N_VALUES   = [100, 500]
const K_VALUES   = [4, 8, 16]
const T_MAX      = 5000
const N_TRIALS   = 50
const BASE_SEED  = 9000

const BEH_THRESH = 0.95
const BEH_WINDOW = 50

# Topology configurations: (label, topology_symbol, p_values)
# For non-smallworld topologies, p is unused
const TOPO_CONFIGS = [
    (:complete,    :complete,   [0.0]),
    (:roundrobin,  :roundrobin, [0.0]),
    (:ring,        :ring,       [0.0]),
    (:sw_p001,     :smallworld, [0.01]),
    (:sw_p01,      :smallworld, [0.1]),
    (:sw_p05,      :smallworld, [0.5]),
    (:scalefree,   :scalefree,  [0.0]),
]

# ══════════════════════════════════════════════════════════════
# Run single trial
# ══════════════════════════════════════════════════════════════

function run_network_trial(N::Int, topology::Symbol, k::Int, p::Float64, seed::Int)
    params = SimulationParams(
        N = N, T = T_MAX, seed = seed,
        w_base = 2, w_max = 6,
        enable_normative = true,
        V = 0, Phi = 0.0,
        network_topology = topology,
        network_degree = k,
    )

    agents, ws, history, rng = initialize(params)
    probes = ProbeSet()
    init_probes!(probes, params, rng)

    # Generate network (nothing for complete graph)
    network = generate_network(topology, N, k; p=p, rng=rng)

    tick_count = 0
    beh_counter = 0
    beh_converged = false
    beh_conv_tick = 0

    for t in 1:T_MAX
        run_tick!(t, agents, ws, history, tick_count, params, rng, probes; network=network)
        tick_count += 1

        # Behavioral convergence check
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

        # Early exit after convergence stabilizes
        if beh_converged && t >= beh_conv_tick + 200
            break
        end
    end

    final = history[tick_count]

    return (
        converged     = beh_converged,
        conv_tick     = beh_conv_tick,
        total_ticks   = tick_count,
        final_coord   = final.coordination_rate,
        final_majority = max(final.fraction_A, 1.0 - final.fraction_A),
        final_cryst   = final.num_crystallised,
        final_norm_str = final.mean_norm_strength,
        final_belief_err = final.belief_error,
    )
end

# ══════════════════════════════════════════════════════════════
# Main sweep
# ══════════════════════════════════════════════════════════════

function run_sweep()
    rows = []
    lk = ReentrantLock()

    total_cells = length(N_VALUES) * length(TOPO_CONFIGS) * length(K_VALUES)
    cell_count = Threads.Atomic{Int}(0)

    for N in N_VALUES
        println("=" ^ 70)
        println("N = $N")
        println("=" ^ 70)

        for (topo_label, topo_sym, p_vals) in TOPO_CONFIGS
            p = p_vals[1]

            for k in K_VALUES
                # Skip topologies that ignore k (complete, roundrobin) for redundant k values
                if topo_sym in (:complete, :roundrobin) && k != K_VALUES[1]
                    continue
                end

                # Skip k > N-1 (can't have more neighbors than agents)
                if k >= N
                    continue
                end

                Threads.atomic_add!(cell_count, 1)
                label = topo_sym == :complete ? "complete" : "$(topo_label)_k$(k)"
                println("  [$( cell_count[])] $label × $N_TRIALS trials (N=$N)...")

                Threads.@threads for trial in 1:N_TRIALS
                    seed = hash((BASE_SEED, N, topo_label, k, trial)) % typemax(Int)
                    result = run_network_trial(N, topo_sym, k, p, Int(seed))

                    lock(lk) do
                        push!(rows, (
                            N              = N,
                            topology       = string(topo_label),
                            degree_k       = k,
                            rewire_p       = p,
                            trial          = trial,
                            converged      = result.converged,
                            conv_tick      = result.conv_tick,
                            total_ticks    = result.total_ticks,
                            final_coord    = result.final_coord,
                            final_majority = result.final_majority,
                            final_cryst    = result.final_cryst,
                            final_norm_str = result.final_norm_str,
                            final_belief_err = result.final_belief_err,
                        ))
                    end
                end
            end
        end
    end

    outpath = joinpath(@__DIR__, "..", "results", "network_topology_sweep.csv")
    CSV.write(outpath, rows)
    println("\nSaved → $outpath")

    return rows
end

# ══════════════════════════════════════════════════════════════
# Console summary
# ══════════════════════════════════════════════════════════════

function print_summary(rows)
    println("\n" * "=" ^ 90)
    println("NETWORK TOPOLOGY SWEEP — SUMMARY")
    println("=" ^ 90)

    for N in N_VALUES
        nrows = [r for r in rows if r.N == N]
        println("\n── N = $N ──")
        println(rpad("Topology", 20), rpad("k", 6), rpad("Conv%", 10),
                rpad("Mean tick", 12), rpad("Med tick", 12), rpad("Final coord", 14))
        println("-" ^ 74)

        # Group by topology × k
        seen = Set{String}()
        for (topo_label, _, _) in TOPO_CONFIGS
            for k in K_VALUES
                key = "$(topo_label)_$(k)"
                key in seen && continue
                push!(seen, key)

                sub = [r for r in nrows if r.topology == string(topo_label) && r.degree_k == k]
                isempty(sub) && continue

                n_conv = count(r -> r.converged, sub)
                conv_pct = round(100 * n_conv / length(sub), digits=1)
                conv_ticks = [r.conv_tick for r in sub if r.converged]
                mean_t = isempty(conv_ticks) ? NaN : round(mean(conv_ticks), digits=1)
                med_t  = isempty(conv_ticks) ? NaN : round(median(conv_ticks), digits=1)
                mean_coord = round(mean(r.final_coord for r in sub), digits=4)

                label = string(topo_label)
                mean_str = isnan(mean_t) ? "—" : string(mean_t)
                med_str  = isnan(med_t)  ? "—" : string(med_t)
                println(rpad(label, 20), rpad(k, 6), rpad("$(conv_pct)%", 10),
                        rpad(mean_str, 12), rpad(med_str, 12), rpad(mean_coord, 14))
            end
        end
    end
end

# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

function main()
    mkpath(joinpath(@__DIR__, "..", "results"))
    rows = run_sweep()
    print_summary(rows)
    println("\n\nDone.")
end

main()
