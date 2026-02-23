#!/usr/bin/env julia
#
# Diagnostic: trace the normative cascade mechanism tick-by-tick
# Runs a single Both (Dynamic+Norm) trial and logs per-tick internal state
# broken down by agent subgroup: uncrystallized / crystallized-to-A / crystallized-to-B
#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DualMemoryABM
using CSV
using Statistics

const N_AGENTS = 100
const T_MAX = 200
const SEED = 42

function run_diagnostic()
    params = SimulationParams(
        N = N_AGENTS, T = T_MAX, seed = SEED,
        w_base = 2, w_max = 6,
        enable_normative = true,
        V = 0, Phi = 0.0,
    )
    agents, ws, history, rng = initialize(params)
    probes = ProbeSet()
    init_probes!(probes, params, rng)
    tick_count = 0

    rows = []

    # Track previous crystal state to detect transitions
    prev_norms = [agents[i].r for i in 1:N_AGENTS]

    for t in 1:T_MAX
        run_tick!(t, agents, ws, history, tick_count, params, rng, probes)
        tick_count += 1

        # ── Classify agents ──
        ids_uncr = Int[]   # uncrystallized
        ids_crA  = Int[]   # crystallized to A
        ids_crB  = Int[]   # crystallized to B

        for i in 1:N_AGENTS
            if agents[i].r == DualMemoryABM.NO_NORM
                push!(ids_uncr, i)
            elseif agents[i].r == DualMemoryABM.STRATEGY_A
                push!(ids_crA, i)
            else
                push!(ids_crB, i)
            end
        end

        n_uncr = length(ids_uncr)
        n_crA  = length(ids_crA)
        n_crB  = length(ids_crB)

        # ── Transitions this tick ──
        new_cryst_A = 0
        new_cryst_B = 0
        dissolutions = 0
        for i in 1:N_AGENTS
            prev = prev_norms[i]
            curr = agents[i].r
            if prev == DualMemoryABM.NO_NORM && curr == DualMemoryABM.STRATEGY_A
                new_cryst_A += 1
            elseif prev == DualMemoryABM.NO_NORM && curr == DualMemoryABM.STRATEGY_B
                new_cryst_B += 1
            elseif prev != DualMemoryABM.NO_NORM && curr == DualMemoryABM.NO_NORM
                dissolutions += 1
            end
            prev_norms[i] = curr
        end

        # ── Per-group statistics ──
        # b_exp_A
        bexp_uncr = n_uncr > 0 ? mean(agents[i].b_exp_A for i in ids_uncr) : NaN
        bexp_crA  = n_crA > 0  ? mean(agents[i].b_exp_A for i in ids_crA) : NaN
        bexp_crB  = n_crB > 0  ? mean(agents[i].b_exp_A for i in ids_crB) : NaN
        bexp_all  = mean(agents[i].b_exp_A for i in 1:N_AGENTS)

        # b_eff_A
        beff_uncr = n_uncr > 0 ? mean(agents[i].b_eff_A for i in ids_uncr) : NaN
        beff_crA  = n_crA > 0  ? mean(agents[i].b_eff_A for i in ids_crA) : NaN
        beff_crB  = n_crB > 0  ? mean(agents[i].b_eff_A for i in ids_crB) : NaN
        beff_all  = mean(agents[i].b_eff_A for i in 1:N_AGENTS)

        # confidence
        C_uncr = n_uncr > 0 ? mean(agents[i].C for i in ids_uncr) : NaN
        C_crA  = n_crA > 0  ? mean(agents[i].C for i in ids_crA) : NaN
        C_crB  = n_crB > 0  ? mean(agents[i].C for i in ids_crB) : NaN
        C_all  = mean(agents[i].C for i in 1:N_AGENTS)

        # sigma (norm strength)
        sigma_crA = n_crA > 0 ? mean(agents[i].sigma for i in ids_crA) : NaN
        sigma_crB = n_crB > 0 ? mean(agents[i].sigma for i in ids_crB) : NaN

        # anomaly count
        anom_crA = n_crA > 0 ? mean(agents[i].a for i in ids_crA) : NaN
        anom_crB = n_crB > 0 ? mean(agents[i].a for i in ids_crB) : NaN

        # compliance
        compl_crA = n_crA > 0 ? mean(agents[i].compliance for i in ids_crA) : NaN
        compl_crB = n_crB > 0 ? mean(agents[i].compliance for i in ids_crB) : NaN

        # DDM evidence for uncrystallized
        e_uncr = n_uncr > 0 ? mean(agents[i].e for i in ids_uncr) : NaN
        e_uncr_abs = n_uncr > 0 ? mean(abs(agents[i].e) for i in ids_uncr) : NaN

        # actions this tick
        m = history[tick_count]

        push!(rows, (
            tick = t,
            # Population level
            fraction_A = m.fraction_A,
            coordination = m.coordination_rate,
            # Group sizes
            n_uncr = n_uncr,
            n_cryst_A = n_crA,
            n_cryst_B = n_crB,
            # Transitions
            new_cryst_A = new_cryst_A,
            new_cryst_B = new_cryst_B,
            dissolutions = dissolutions,
            # b_exp_A per group
            bexp_all = round(bexp_all, digits=4),
            bexp_uncr = round(bexp_uncr, digits=4),
            bexp_crA = round(bexp_crA, digits=4),
            bexp_crB = round(bexp_crB, digits=4),
            # b_eff_A per group
            beff_all = round(beff_all, digits=4),
            beff_uncr = round(beff_uncr, digits=4),
            beff_crA = round(beff_crA, digits=4),
            beff_crB = round(beff_crB, digits=4),
            # confidence
            C_all = round(C_all, digits=4),
            C_uncr = round(C_uncr, digits=4),
            C_crA = round(C_crA, digits=4),
            C_crB = round(C_crB, digits=4),
            # norm strength
            sigma_crA = round(sigma_crA, digits=4),
            sigma_crB = round(sigma_crB, digits=4),
            # anomaly
            anom_crA = round(anom_crA, digits=2),
            anom_crB = round(anom_crB, digits=2),
            # compliance
            compl_crA = round(compl_crA, digits=4),
            compl_crB = round(compl_crB, digits=4),
            # DDM
            e_uncr_mean = round(e_uncr, digits=3),
            e_uncr_abs_mean = round(e_uncr_abs, digits=3),
        ))
    end

    outpath = joinpath(@__DIR__, "..", "results", "cascade_diagnostic.csv")
    CSV.write(outpath, rows)
    println("Saved → $outpath")

    # ── Console: key moments ──
    println("\n", "=" ^ 120)
    println("CASCADE DIAGNOSTIC (N=$N_AGENTS, seed=$SEED, Dynamic+Norm, V=0, Phi=0.0)")
    println("=" ^ 120)

    println("\n┌─ Phase timeline ─────────────────────────────────────────────────────────────────────┐")
    println("│ tick │ frac_A │coord│ uncr│ crA│ crB│ +crA│ +crB│ diss│ bexp_all│beff_all│ C_all │σ_crA│σ_crB│anom_A│anom_B│")
    println("├──────┼────────┼─────┼─────┼────┼────┼─────┼─────┼─────┼────────┼────────┼───────┼─────┼─────┼──────┼──────┤")

    for r in rows
        if r.tick <= 30 || r.tick % 5 == 0 || r.tick <= 80
            sa = isnan(r.sigma_crA) ? "  -- " : lpad(string(round(r.sigma_crA, digits=2)), 5)
            sb = isnan(r.sigma_crB) ? "  -- " : lpad(string(round(r.sigma_crB, digits=2)), 5)
            aa = isnan(r.anom_crA) ? "  -- " : lpad(string(round(r.anom_crA, digits=1)), 5)
            ab = isnan(r.anom_crB) ? "  -- " : lpad(string(round(r.anom_crB, digits=1)), 5)

            println("│ ", lpad(r.tick, 4), " │ ",
                    lpad(r.fraction_A, 5), " │",
                    lpad(r.coordination, 4), " │",
                    lpad(r.n_uncr, 4), " │",
                    lpad(r.n_cryst_A, 3), " │",
                    lpad(r.n_cryst_B, 3), " │",
                    lpad(r.new_cryst_A, 4), " │",
                    lpad(r.new_cryst_B, 4), " │",
                    lpad(r.dissolutions, 4), " │",
                    lpad(r.bexp_all, 7), " │",
                    lpad(r.beff_all, 7), " │",
                    lpad(r.C_all, 6), " │",
                    sa, "│",
                    sb, "│",
                    aa, " │",
                    ab, " │")
        end
    end
    println("└──────┴────────┴─────┴─────┴────┴────┴─────┴─────┴─────┴────────┴────────┴───────┴─────┴─────┴──────┴──────┘")

    # ── Key transition moments ──
    println("\n── Per-group b_eff_A at key ticks ──\n")
    println(rpad("tick", 6), rpad("b_eff uncr", 12), rpad("b_eff crA", 12), rpad("b_eff crB", 12),
            rpad("compl crA", 12), rpad("compl crB", 12))
    println("-" ^ 66)
    for r in rows
        if r.tick in [1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 100]
            bu = isnan(r.beff_uncr) ? "--" : string(round(r.beff_uncr, digits=3))
            ba = isnan(r.beff_crA) ? "--" : string(round(r.beff_crA, digits=3))
            bb = isnan(r.beff_crB) ? "--" : string(round(r.beff_crB, digits=3))
            ca = isnan(r.compl_crA) ? "--" : string(round(r.compl_crA, digits=3))
            cb = isnan(r.compl_crB) ? "--" : string(round(r.compl_crB, digits=3))
            println(rpad(r.tick, 6), rpad(bu, 12), rpad(ba, 12), rpad(bb, 12), rpad(ca, 12), rpad(cb, 12))
        end
    end
end

run_diagnostic()
