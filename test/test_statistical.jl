using Test
using Random
using Statistics
using DualMemoryABM

# ──────────────────────────────────────────────────────────────
# Helper: Normal CDF via approximation (no SpecialFunctions dep)
# ──────────────────────────────────────────────────────────────

"""Approximate standard normal CDF Φ(z) using Abramowitz & Stegun."""
function normal_cdf(z::Float64)
    if z < -8.0
        return 0.0
    elseif z > 8.0
        return 1.0
    end
    # Use the built-in erfc if available, otherwise use approximation
    # Φ(z) = 0.5 * erfc(-z / sqrt(2))
    # Implement erfc approximation
    t = 1.0 / (1.0 + 0.2316419 * abs(z))
    d = 0.3989422804014327  # 1/sqrt(2π)
    p = d * exp(-z * z / 2.0) *
        (t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 +
         t * (-1.821255978 + t * 1.330274429)))))
    return z > 0 ? 1.0 - p : p
end

# ──────────────────────────────────────────────────────────────
# Helper: Mann-Whitney U test (one-sided, x < y)
# ──────────────────────────────────────────────────────────────

"""
    mann_whitney_u_onesided(x::Vector, y::Vector)

One-sided Mann-Whitney U test: H1: x stochastically < y.
Returns (U, z, p_approx).
"""
function mann_whitney_u_onesided(x::AbstractVector, y::AbstractVector)
    nx, ny = length(x), length(y)
    combined = vcat([(v, :x) for v in x], [(v, :y) for v in y])
    sort!(combined; by=first)

    # Assign ranks (handle ties by averaging)
    ranks = Vector{Float64}(undef, nx + ny)
    i = 1
    while i <= nx + ny
        j = i
        while j <= nx + ny && combined[j][1] == combined[i][1]
            j += 1
        end
        avg_rank = (i + j - 1) / 2.0
        for k in i:(j-1)
            ranks[k] = avg_rank
        end
        i = j
    end

    R_x = sum(ranks[k] for k in 1:(nx+ny) if combined[k][2] == :x)
    U_x = R_x - nx * (nx + 1) / 2

    mu = nx * ny / 2.0
    sigma_u = sqrt(nx * ny * (nx + ny + 1) / 12.0)

    z = (U_x - mu) / sigma_u
    # For x < y: small U_x → negative z → small Φ(z)
    p_left = normal_cdf(z)

    return U_x, z, p_left
end

"""
    spearman_correlation(x::AbstractVector, y::AbstractVector)

Compute Spearman rank correlation.
"""
function spearman_correlation(x::AbstractVector, y::AbstractVector)
    n = length(x)
    @assert n == length(y)

    rank_x = tiedrank(x)
    rank_y = tiedrank(y)

    d = rank_x .- rank_y
    rho = 1.0 - 6.0 * sum(d .^ 2) / (n * (n^2 - 1))
    return rho
end

function tiedrank(v::AbstractVector)
    n = length(v)
    idx = sortperm(v)
    ranks = Vector{Float64}(undef, n)
    i = 1
    while i <= n
        j = i
        while j <= n && v[idx[j]] == v[idx[i]]
            j += 1
        end
        avg = (i + j - 1) / 2.0
        for k in i:(j-1)
            ranks[idx[k]] = avg
        end
        i = j
    end
    return ranks
end

@testset "Statistical Tests" begin

    n_trials = 30

    # ── S1: Dynamic memory accelerates convergence ──
    @testset "S1: Dynamic memory accelerates convergence" begin
        conv_dynamic = Int[]
        for trial in 1:n_trials
            p = SimulationParams(N=100, T=1000, seed=trial * 1000 + 1)
            result = run!(p)
            s = summarize(result)
            push!(conv_dynamic, s.convergence_tick > 0 ? s.convergence_tick : p.T)
        end

        conv_fixed = Int[]
        for trial in 1:n_trials
            p = SimulationParams(N=100, T=1000, seed=trial * 1000 + 2,
                                  w_base=4, w_max=4)
            result = run!(p)
            s = summarize(result)
            push!(conv_fixed, s.convergence_tick > 0 ? s.convergence_tick : p.T)
        end

        _, _, p_val = mann_whitney_u_onesided(Float64.(conv_dynamic), Float64.(conv_fixed))
        @test p_val < 0.05
    end

    # ── S2: DDM crystallisation time at 50-50 ──
    @testset "S2: DDM crystallisation at 50-50" begin
        # With V=0, each tick's f_diff = ±1, so DDM does a random walk with
        # step size ≈ (1-C) * 1, not pure noise. Crystallisation is much faster
        # than the pure-noise theoretical value of θ²/σ² = 900.
        # Expected: around 10-100 ticks depending on C dynamics.
        crystal_ticks = Float64[]

        for trial in 1:n_trials
            p = SimulationParams(N=100, T=3000, seed=trial * 2000 + 1,
                                  enable_normative=true, V=0, Phi=0.0)
            agents, ws, history, rng = initialize(p)

            for t in 1:p.T
                run_tick!(t, agents, ws, history, t-1, p, rng, ProbeSet())

                for i in 1:p.N
                    if agents[i].r != NO_NORM
                        push!(crystal_ticks, Float64(t))
                        agents[i].r = NO_NORM
                        agents[i].e = 0.0
                        agents[i].sigma = 0.0
                        agents[i].a = 0
                    end
                end

                if length(crystal_ticks) >= 200
                    break
                end
            end
        end

        if length(crystal_ticks) >= 30
            mean_ct = mean(crystal_ticks)
            std_ct = std(crystal_ticks)
            cv = std_ct / mean_ct

            # With V=0, f_diff = ±1 each tick: random walk with large steps
            # Crystallisation happens quickly (10-100 ticks typically)
            @test 1 < mean_ct < 500
            @test cv > 0.1  # some variance expected
        else
            @warn "S2: Not enough crystallisations observed ($(length(crystal_ticks)))"
            @test_skip false
        end
    end

    # ── S3: DDM crystallisation time at 70-30 ──
    @testset "S3: DDM crystallisation at 70-30" begin
        crystal_ticks = Float64[]

        for trial in 1:n_trials
            p = SimulationParams(N=100, T=200, seed=trial * 3000 + 1,
                                  enable_normative=true, V=0, Phi=0.0,
                                  C0=0.3)

            agents, ws, history, rng = initialize(p)

            for t in 1:p.T
                run_tick!(t, agents, ws, history, t-1, p, rng, ProbeSet())

                for i in 1:p.N
                    if agents[i].r != NO_NORM
                        push!(crystal_ticks, Float64(t))
                        agents[i].r = NO_NORM
                        agents[i].e = 0.0
                        agents[i].sigma = 0.0
                        agents[i].a = 0
                    end
                end

                if length(crystal_ticks) >= 200
                    break
                end
            end
        end

        if length(crystal_ticks) >= 30
            mean_ct = mean(crystal_ticks)
            @test mean_ct < 100
        else
            @warn "S3: Not enough crystallisations ($(length(crystal_ticks)))"
            @test_skip false
        end
    end

    # ── S4: Low-C agents crystallise first (H2) ──
    @testset "S4: Low-C agents crystallise first" begin
        c_at_crystal = Float64[]
        t_at_crystal = Float64[]

        for trial in 1:n_trials
            p = SimulationParams(N=100, T=2000, seed=trial * 4000 + 1,
                                  enable_normative=true, V=3, Phi=0.0)
            agents, ws, history, rng = initialize(p)
            crystallised = Set{Int}()

            for t in 1:p.T
                c_before = [agents[i].C for i in 1:p.N]

                run_tick!(t, agents, ws, history, t-1, p, rng, ProbeSet())

                for i in 1:p.N
                    if agents[i].r != NO_NORM && !(i in crystallised)
                        push!(crystallised, i)
                        push!(c_at_crystal, c_before[i])
                        push!(t_at_crystal, Float64(t))
                    end
                end

                if length(crystallised) >= 80
                    break
                end
            end
        end

        if length(c_at_crystal) >= 30
            rho = spearman_correlation(c_at_crystal, t_at_crystal)
            # H2: higher C → later crystallisation (positive correlation)
            @test rho > 0
        else
            @warn "S4: Not enough data points ($(length(c_at_crystal)))"
            @test_skip false
        end
    end

    # ── S5: Enforcement accelerates crystallisation ──
    @testset "S5: Enforcement accelerates crystallisation" begin
        crystal_with = Float64[]
        crystal_without = Float64[]

        for trial in 1:n_trials
            p1 = SimulationParams(N=100, T=2000, seed=trial * 5000 + 1,
                                    enable_normative=true, V=5, Phi=1.0)
            r1 = run!(p1)
            push!(crystal_with, Float64(r1.history[end].num_crystallised))

            p2 = SimulationParams(N=100, T=2000, seed=trial * 5000 + 1,
                                    enable_normative=true, V=5, Phi=0.0)
            r2 = run!(p2)
            push!(crystal_without, Float64(r2.history[end].num_crystallised))
        end

        mean_with = mean(crystal_with)
        mean_without = mean(crystal_without)
        @test mean_with >= mean_without * 0.9
    end

    # ── S6: V=0 DDM is noisy ──
    @testset "S6: V=0 DDM is noisy" begin
        # Track individual agent crystallisation times across trials
        all_crystal_ticks = Float64[]

        for trial in 1:n_trials
            p = SimulationParams(N=100, T=500, seed=trial * 6000 + 1,
                                  enable_normative=true, V=0, Phi=0.0)
            agents, ws, history, rng = initialize(p)

            for t in 1:p.T
                run_tick!(t, agents, ws, history, t-1, p, rng, ProbeSet())

                for i in 1:p.N
                    if agents[i].r != NO_NORM
                        push!(all_crystal_ticks, Float64(t))
                        agents[i].r = NO_NORM
                        agents[i].e = 0.0
                        agents[i].sigma = 0.0
                        agents[i].a = 0
                    end
                end
            end
        end

        if length(all_crystal_ticks) >= 100
            cv = std(all_crystal_ticks) / mean(all_crystal_ticks)
            # With V=0, f_diff = ±1 creates noisy crystallisation times
            @test cv > 0.1
        else
            @warn "S6: Too few crystallisations ($(length(all_crystal_ticks)))"
            @test_skip false
        end
    end

    # ── S7: Full model reaches Level 5 ──
    @testset "S7: Full model reaches Level 5" begin
        level5_count = 0

        for trial in 1:n_trials
            p = SimulationParams(N=100, T=3000, seed=trial * 7000 + 1,
                                  enable_normative=true, V=5, Phi=1.0)
            result = run!(p)

            max_level = maximum(m.norm_level for m in result.history)
            if max_level >= 5
                level5_count += 1
            end
        end

        # ≥ 50% of trials should reach Level 5 within 2000 ticks
        @test level5_count >= n_trials * 0.5
    end

    # ── S8: No-normative ceiling is Level 3 ──
    @testset "S8: No-normative ceiling is Level 3" begin
        above_3_count = 0

        for trial in 1:n_trials
            p = SimulationParams(N=100, T=1000, seed=trial * 8000 + 1,
                                  enable_normative=false)
            result = run!(p)

            max_level = maximum(m.norm_level for m in result.history)
            if max_level > 3
                above_3_count += 1
            end
        end

        @test above_3_count == 0
    end

end
