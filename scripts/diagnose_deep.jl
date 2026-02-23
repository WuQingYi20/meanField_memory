#!/usr/bin/env julia
#
# Deep diagnostic: compare Both vs Exp_only + individual agent tracking
#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DualMemoryABM
using CSV
using Statistics

const N_AGENTS = 100
const T_MAX = 300

# ══════════════════════════════════════════════════════════════
# Part 1: Side-by-side Both vs Exp_only (same seed)
# ══════════════════════════════════════════════════════════════

function run_condition(seed::Int; enable_norm::Bool, label::String)
    params = SimulationParams(
        N = N_AGENTS, T = T_MAX, seed = seed,
        w_base = 2, w_max = 6,
        enable_normative = enable_norm,
        V = 0, Phi = 0.0,
    )
    agents, ws, history, rng = initialize(params)
    probes = ProbeSet()
    init_probes!(probes, params, rng)
    tick_count = 0

    rows = []

    for t in 1:T_MAX
        run_tick!(t, agents, ws, history, tick_count, params, rng, probes)
        tick_count += 1
        m = history[tick_count]

        n_crA = count(i -> agents[i].r == DualMemoryABM.STRATEGY_A, 1:N_AGENTS)
        n_crB = count(i -> agents[i].r == DualMemoryABM.STRATEGY_B, 1:N_AGENTS)

        bexp_all = mean(agents[i].b_exp_A for i in 1:N_AGENTS)
        beff_all = mean(agents[i].b_eff_A for i in 1:N_AGENTS)
        C_all = mean(agents[i].C for i in 1:N_AGENTS)

        push!(rows, (
            condition = label,
            tick = t,
            fraction_A = m.fraction_A,
            coordination = m.coordination_rate,
            n_cryst_A = n_crA,
            n_cryst_B = n_crB,
            bexp_all = round(bexp_all, digits=4),
            beff_all = round(beff_all, digits=4),
            C_all = round(C_all, digits=4),
        ))
    end
    return rows
end

# ══════════════════════════════════════════════════════════════
# Part 2: Individual agent lifecycle in Both condition
# ══════════════════════════════════════════════════════════════

function run_agent_tracking(seed::Int)
    params = SimulationParams(
        N = N_AGENTS, T = T_MAX, seed = seed,
        w_base = 2, w_max = 6,
        enable_normative = true,
        V = 0, Phi = 0.0,
    )
    agents, ws, history, rng = initialize(params)
    probes = ProbeSet()
    init_probes!(probes, params, rng)
    tick_count = 0

    # Track state transitions for every agent
    # Format: (agent_id, tick, event, from_state, to_state)
    transitions = []
    prev_norms = fill(DualMemoryABM.NO_NORM, N_AGENTS)

    # Per-tick agent snapshot (for selected agents)
    agent_rows = []

    for t in 1:T_MAX
        run_tick!(t, agents, ws, history, tick_count, params, rng, probes)
        tick_count += 1

        for i in 1:N_AGENTS
            curr = agents[i].r
            prev = prev_norms[i]

            if prev != curr
                from_str = prev == DualMemoryABM.NO_NORM ? "uncr" :
                           prev == DualMemoryABM.STRATEGY_A ? "crA" : "crB"
                to_str   = curr == DualMemoryABM.NO_NORM ? "uncr" :
                           curr == DualMemoryABM.STRATEGY_A ? "crA" : "crB"
                push!(transitions, (
                    agent = i, tick = t,
                    from = from_str, to = to_str,
                    bexp_A = round(agents[i].b_exp_A, digits=4),
                    sigma = round(agents[i].sigma, digits=4),
                    e = round(agents[i].e, digits=3),
                    C = round(agents[i].C, digits=4),
                ))
            end
            prev_norms[i] = curr
        end

        # Snapshot every agent every tick (for detailed analysis)
        for i in 1:N_AGENTS
            push!(agent_rows, (
                tick = t,
                agent = i,
                r = agents[i].r == DualMemoryABM.NO_NORM ? -1 :
                    agents[i].r == DualMemoryABM.STRATEGY_A ? 1 : 0,
                bexp_A = round(agents[i].b_exp_A, digits=4),
                beff_A = round(agents[i].b_eff_A, digits=4),
                sigma = round(agents[i].sigma, digits=4),
                a = agents[i].a,
                e = round(agents[i].e, digits=3),
                C = round(agents[i].C, digits=4),
            ))
        end
    end

    return transitions, agent_rows
end

# ══════════════════════════════════════════════════════════════
# Part 3: Multi-seed cascade trigger analysis
# ══════════════════════════════════════════════════════════════

function find_cascade_trigger(seeds::Vector{Int})
    rows = []

    for seed in seeds
        params = SimulationParams(
            N = N_AGENTS, T = T_MAX, seed = seed,
            w_base = 2, w_max = 6,
            enable_normative = true,
            V = 0, Phi = 0.0,
        )
        agents, ws, history, rng = initialize(params)
        probes = ProbeSet()
        init_probes!(probes, params, rng)
        tick_count = 0

        prev_norms = fill(DualMemoryABM.NO_NORM, N_AGENTS)

        # Track cascade start: first tick where |crA - crB| / (crA + crB) > 0.3
        # and it's sustained (never reverses)
        cascade_tick = 0
        cascade_ratio = 0.0
        cascade_bexp = 0.0
        cascade_frac_A = 0.0
        peak_symmetric_tick = 0  # last tick where |crA - crB| <= 2

        conv_tick = 0
        beh_counter = 0

        for t in 1:T_MAX
            run_tick!(t, agents, ws, history, tick_count, params, rng, probes)
            tick_count += 1
            m = history[tick_count]

            n_crA = count(i -> agents[i].r == DualMemoryABM.STRATEGY_A, 1:N_AGENTS)
            n_crB = count(i -> agents[i].r == DualMemoryABM.STRATEGY_B, 1:N_AGENTS)
            n_cr = n_crA + n_crB
            bexp_all = mean(agents[i].b_exp_A for i in 1:N_AGENTS)

            # Track symmetric phase
            if n_cr > 0 && abs(n_crA - n_crB) <= 2
                peak_symmetric_tick = t
            end

            # Detect cascade start: first tick minority norm = 0
            majority = max(m.fraction_A, 1.0 - m.fraction_A)
            minority_norm = min(n_crA, n_crB)

            if cascade_tick == 0 && n_cr > 20 && minority_norm == 0
                cascade_tick = t
                cascade_ratio = n_cr > 0 ? n_crA / n_cr : 0.5
                cascade_bexp = bexp_all
                cascade_frac_A = m.fraction_A
            end

            # Behavioral convergence
            if majority >= 0.95
                beh_counter += 1
                if conv_tick == 0 && beh_counter >= 50
                    conv_tick = t - 49
                end
            else
                beh_counter = 0
            end

            # Also track tipping point: when does |crA - crB| first exceed 10?
        end

        # Find tipping point (|crA - crB| > 10 sustained)
        # Re-run to find this... or use stored data
        # Actually let's find it from agent state at peak_symmetric_tick

        push!(rows, (
            seed = seed,
            peak_symmetric_tick = peak_symmetric_tick,
            cascade_complete_tick = cascade_tick,
            cascade_duration = cascade_tick > 0 ? cascade_tick - peak_symmetric_tick : 0,
            cascade_bexp = round(cascade_bexp, digits=3),
            cascade_frac_A = round(cascade_frac_A, digits=3),
            conv_tick = conv_tick,
        ))
    end

    return rows
end

# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

function main()
    mkpath(joinpath(@__DIR__, "..", "results"))
    seed = 42

    # ── Part 1: Side-by-side comparison ──
    println("Part 1: Both vs Exp_only (seed=$seed)")
    rows_both = run_condition(seed, enable_norm=true, label="Both")
    rows_exp  = run_condition(seed, enable_norm=false, label="Exp_only")
    all_comparison = vcat(rows_both, rows_exp)
    CSV.write(joinpath(@__DIR__, "..", "results", "cascade_comparison.csv"), all_comparison)

    # Print side-by-side at key ticks
    println("\n", "=" ^ 100)
    println("SIDE-BY-SIDE: Both vs Exp_only (seed=$seed)")
    println("=" ^ 100)
    println(rpad("tick", 6),
            "│ ", rpad("Both frac_A", 12), rpad("coord", 8), rpad("crA", 5), rpad("crB", 5), rpad("bexp", 8),
            "│ ", rpad("Exp frac_A", 12), rpad("coord", 8), rpad("bexp", 8))
    println("-" ^ 95)

    for t in [1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,90,100,
              120,150,200,250,300]
        rb = rows_both[t]
        re = rows_exp[min(t, length(rows_exp))]
        println(rpad(t, 6), "│ ",
                rpad(rb.fraction_A, 12), rpad(rb.coordination, 8),
                rpad(rb.n_cryst_A, 5), rpad(rb.n_cryst_B, 5), rpad(rb.bexp_all, 8),
                "│ ",
                rpad(re.fraction_A, 12), rpad(re.coordination, 8), rpad(re.bexp_all, 8))
    end

    # ── Part 2: Agent lifecycle tracking ──
    println("\n\nPart 2: Individual agent transitions (seed=$seed)")
    transitions, agent_rows = run_agent_tracking(seed)

    CSV.write(joinpath(@__DIR__, "..", "results", "cascade_transitions.csv"), transitions)
    CSV.write(joinpath(@__DIR__, "..", "results", "cascade_agent_snapshots.csv"), agent_rows)

    # Analyze B->dissolution->re-crystallize pathways
    println("\n", "=" ^ 90)
    println("AGENT LIFECYCLE: B-crystallized -> dissolution -> re-crystallization")
    println("=" ^ 90)

    # Find agents that went crB -> uncr -> crA
    b_agents = Dict{Int, Vector{NamedTuple}}()
    for tr in transitions
        if !haskey(b_agents, tr.agent)
            b_agents[tr.agent] = []
        end
        push!(b_agents[tr.agent], tr)
    end

    println("\nAgents that followed B -> dissolution -> re-crystallize-to-A pathway:")
    println(rpad("agent", 7), rpad("cryst_B_tick", 13), rpad("dissolve_tick", 14),
            rpad("re_cryst_A_tick", 16), rpad("B_duration", 12), rpad("uncr_duration", 14),
            rpad("bexp_at_recryst", 16))
    println("-" ^ 92)

    pathway_count = 0
    b_durations = Int[]
    uncr_durations = Int[]
    recryst_bexp = Float64[]

    for (aid, trs) in sort(collect(b_agents))
        # Look for pattern: uncr->crB, then crB->uncr, then uncr->crA
        for i in 1:length(trs)
            if trs[i].to == "crB"
                crB_tick = trs[i].tick
                # Find next dissolution
                for j in (i+1):length(trs)
                    if trs[j].from == "crB" && trs[j].to == "uncr"
                        diss_tick = trs[j].tick
                        # Find next re-crystallization
                        for k in (j+1):length(trs)
                            if trs[k].from == "uncr" && trs[k].to == "crA"
                                recryst_tick = trs[k].tick
                                pathway_count += 1
                                b_dur = diss_tick - crB_tick
                                u_dur = recryst_tick - diss_tick
                                push!(b_durations, b_dur)
                                push!(uncr_durations, u_dur)
                                push!(recryst_bexp, trs[k].bexp_A)

                                if pathway_count <= 20
                                    println(rpad(aid, 7), rpad(crB_tick, 13), rpad(diss_tick, 14),
                                            rpad(recryst_tick, 16), rpad(b_dur, 12), rpad(u_dur, 14),
                                            rpad(round(trs[k].bexp_A, digits=3), 16))
                                end
                                break
                            elseif trs[k].from == "uncr" && trs[k].to == "crB"
                                break  # re-crystallized to B again
                            end
                        end
                        break
                    end
                end
            end
        end
    end

    println("\nTotal B->uncr->A pathways: $pathway_count")
    if length(b_durations) > 0
        println("B-crystallized duration:  mean=$(round(mean(b_durations), digits=1)), median=$(median(b_durations)), range=[$(minimum(b_durations)), $(maximum(b_durations))]")
        println("Uncrystallized duration:  mean=$(round(mean(uncr_durations), digits=1)), median=$(median(uncr_durations)), range=[$(minimum(uncr_durations)), $(maximum(uncr_durations))]")
        println("b_exp_A at re-cryst to A: mean=$(round(mean(recryst_bexp), digits=3)), range=[$(round(minimum(recryst_bexp), digits=3)), $(round(maximum(recryst_bexp), digits=3))]")
    end

    # Also count agents that went crB -> uncr -> crB (re-crystallized to B again)
    recryst_B_count = 0
    for (aid, trs) in b_agents
        for i in 1:length(trs)
            if trs[i].to == "crB"
                for j in (i+1):length(trs)
                    if trs[j].from == "crB" && trs[j].to == "uncr"
                        for k in (j+1):length(trs)
                            if trs[k].from == "uncr" && trs[k].to == "crB"
                                recryst_B_count += 1
                                break
                            elseif trs[k].from == "uncr" && trs[k].to == "crA"
                                break
                            end
                        end
                        break
                    end
                end
            end
        end
    end
    println("Total B->uncr->B (re-crystallized to same B): $recryst_B_count")

    # ── Part 3: Multi-seed cascade analysis ──
    println("\n\nPart 3: Multi-seed cascade trigger analysis (30 seeds)")
    seeds = collect(1:30)
    trigger_rows = find_cascade_trigger(seeds)
    CSV.write(joinpath(@__DIR__, "..", "results", "cascade_trigger_analysis.csv"), trigger_rows)

    println("\n", "=" ^ 90)
    println("CASCADE TRIGGER ANALYSIS (30 seeds, N=100)")
    println("=" ^ 90)
    println(rpad("seed", 6), rpad("symmetric_end", 15), rpad("cascade_done", 14),
            rpad("duration", 10), rpad("bexp_at_end", 13), rpad("conv_tick", 10))
    println("-" ^ 68)

    for r in trigger_rows
        println(rpad(r.seed, 6), rpad(r.peak_symmetric_tick, 15),
                rpad(r.cascade_complete_tick, 14),
                rpad(r.cascade_duration, 10),
                rpad(r.cascade_bexp, 13),
                rpad(r.conv_tick, 10))
    end

    valid = [r for r in trigger_rows if r.cascade_complete_tick > 0]
    if length(valid) > 0
        durations = [r.cascade_duration for r in valid]
        sym_ticks = [r.peak_symmetric_tick for r in valid]
        conv_ticks = [r.conv_tick for r in valid if r.conv_tick > 0]
        println("\nSummary over $(length(valid)) converged seeds:")
        println("  Symmetric phase ends:    mean=$(round(mean(sym_ticks), digits=1)), median=$(median(sym_ticks))")
        println("  Cascade duration:        mean=$(round(mean(durations), digits=1)), median=$(median(durations))")
        println("  Behavioral convergence:  mean=$(round(mean(conv_ticks), digits=1)), median=$(median(conv_ticks))")
    end

    println("\n\nAll data saved to results/cascade_*.csv")
end

main()
