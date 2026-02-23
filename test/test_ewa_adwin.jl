using Test
using Random
using Statistics
using DualMemoryABM

@testset "EWA-ADWIN Tests" begin

    # ══════════════════════════════════════════════════════════════
    # Deterministic Tests (EA1-EA6)
    # ══════════════════════════════════════════════════════════════

    # ── EA1: Symmetric initialization ──
    @testset "EA1: Symmetric init" begin
        params = EWAADWINParams(N=10, T=100, seed=42)
        agents, ws, history, rng = ewa_adwin_initialize(params)

        for agent in agents
            @test agent.attract_A == 0.0
            @test agent.attract_B == 0.0
            @test agent.prob_A == 0.5
            @test adwin_count(agent.adwin_A) == 0
            @test adwin_count(agent.adwin_B) == 0
        end
    end

    # ── EA2: Both play A → attract_A grows, attract_B stays low ──
    @testset "EA2: Both play A reinforcement" begin
        params = EWAADWINParams(N=2, T=100, seed=42, delta=0.5, lambda=1.0)
        agents, ws, history, rng = ewa_adwin_initialize(params)

        # Manually set up: both agents play A against each other
        ws.pair_i[1] = 1
        ws.pair_j[1] = 2
        ws.perm[1] = 1
        ws.perm[2] = 2
        ws.action[1] = STRATEGY_A
        ws.action[2] = STRATEGY_A
        ws.coordinated[1] = true

        # Run update
        ewa_adwin_update_attractions!(agents, ws, params)

        # Both played A, partner played A:
        # reinf_A = (0.5 + 0.5*1)*1 = 1.0, reinf_B = (0.5 + 0.5*0)*0 = 0.0
        @test agents[1].attract_A ≈ 1.0  # mean of single observation 1.0
        @test agents[1].attract_B ≈ 0.0  # mean of single observation 0.0
        @test agents[1].prob_A > 0.5

        # Run several more ticks with same setup
        for _ in 1:10
            ws.action[1] = STRATEGY_A
            ws.action[2] = STRATEGY_A
            ewa_adwin_update_attractions!(agents, ws, params)
        end

        @test agents[1].attract_A > agents[1].attract_B
        @test agents[1].prob_A > 0.7
    end

    # ── EA3: Logit probability computation ──
    @testset "EA3: Logit probability" begin
        params = EWAADWINParams(N=2, T=100, seed=42)
        agents, _, _, _ = ewa_adwin_initialize(params)

        agent = agents[1]
        agent.attract_A = 1.0
        agent.attract_B = 0.5

        _ewa_adwin_update_prob!(agent, 2.0)
        expected = exp(2.0) / (exp(2.0) + exp(1.0))
        @test agent.prob_A ≈ expected atol=1e-10

        # Equal attractions → prob = 0.5
        agent.attract_A = 5.0
        agent.attract_B = 5.0
        _ewa_adwin_update_prob!(agent, 3.0)
        @test agent.prob_A ≈ 0.5

        # Numerical stability with large values
        agent.attract_A = 100.0
        agent.attract_B = 99.0
        _ewa_adwin_update_prob!(agent, 10.0)
        @test agent.prob_A > 0.999
        @test agent.prob_A <= 1.0
    end

    # ── EA4: Normative fields are zero in metrics ──
    @testset "EA4: Normative fields zero" begin
        params = EWAADWINParams(N=20, T=50, seed=42)
        result = ewa_adwin_run!(params)

        for m in result.history
            @test m.mean_confidence == 0.0
            @test m.num_crystallised == 0
            @test m.mean_norm_strength == 0.0
            @test m.num_enforcements == 0
            @test m.frac_dominant_norm == 0.0
        end
    end

    # ── EA5: Smoke test ──
    @testset "EA5: Smoke test (ewa_adwin_run!)" begin
        params = EWAADWINParams(N=20, T=200, seed=42, lambda=3.0)
        result = ewa_adwin_run!(params)

        @test result.tick_count > 0
        @test result.tick_count <= 200
        @test length(result.history) == result.tick_count
        @test result.probes === nothing

        last_m = result.history[end]
        @test 0.0 <= last_m.fraction_A <= 1.0
        @test 0.0 <= last_m.coordination_rate <= 1.0
    end

    # ── EA6: Parameter validation ──
    @testset "EA6: Parameter validation" begin
        @test_throws ArgumentError validate(EWAADWINParams(N=3))     # odd
        @test_throws ArgumentError validate(EWAADWINParams(N=1))     # too small
        @test_throws ArgumentError validate(EWAADWINParams(delta=-0.1))
        @test_throws ArgumentError validate(EWAADWINParams(delta=1.1))
        @test_throws ArgumentError validate(EWAADWINParams(lambda=0.0))
        @test_throws ArgumentError validate(EWAADWINParams(lambda=-1.0))
        @test_throws ArgumentError validate(EWAADWINParams(adwin_delta=0.0))  # must be > 0
        @test_throws ArgumentError validate(EWAADWINParams(adwin_M=0))
        @test_throws ArgumentError validate(EWAADWINParams(convergence_window=0))

        # Valid edge cases
        @test validate(EWAADWINParams(delta=0.0)) === nothing  # pure RL
        @test validate(EWAADWINParams(delta=1.0)) === nothing  # pure belief
    end

    # ── Additional: summarize() works with EWA-ADWIN ──
    @testset "EWA-ADWIN summarize compatibility" begin
        params = EWAADWINParams(N=20, T=100, seed=42)
        result = ewa_adwin_run!(params)
        ts = summarize(result)

        @test ts isa TrialSummary
        @test ts.total_ticks == result.tick_count
        @test ts.final_num_crystallised == 0
        @test ts.final_mean_norm_strength == 0.0
    end

    # ── Additional: ewa_adwin_first_tick_per_layer ──
    @testset "EWA-ADWIN first_tick_per_layer" begin
        params = EWAADWINParams(N=20, T=200, seed=42, lambda=3.0)
        result = ewa_adwin_run!(params)
        layers = ewa_adwin_first_tick_per_layer(result.history, params.N, params)

        @test layers.crystal == 0  # always 0 for EWA-ADWIN
        @test layers isa NamedTuple{(:behavioral, :belief, :crystal, :all_met)}
    end

    # ── Additional: to_namedtuple round-trip ──
    @testset "EWA-ADWIN to_namedtuple" begin
        params = EWAADWINParams(N=50, delta=0.7, lambda=2.0, adwin_delta=0.05)
        nt = to_namedtuple(params)
        params2 = EWAADWINParams(; nt...)
        @test params2.N == 50
        @test params2.delta == 0.7
        @test params2.lambda == 2.0
        @test params2.adwin_delta == 0.05
    end

    # ══════════════════════════════════════════════════════════════
    # Statistical Tests (ES-A1, ES-A2)
    # ══════════════════════════════════════════════════════════════

    # ── ES-A1: Convergence rate >= 70% ──
    @testset "ES-A1: EWA-ADWIN convergence rate" begin
        n_trials = 30
        n_converged = 0
        for trial in 1:n_trials
            result = ewa_adwin_run!(EWAADWINParams(N=100, T=3000, seed=trial, lambda=5.0))
            ts = summarize(result)
            if ts.converged
                n_converged += 1
            end
        end
        rate = n_converged / n_trials
        @test rate >= 0.70
    end

    # ── ES-A2: Deterministic seeding ──
    @testset "ES-A2: Deterministic seeding" begin
        params = EWAADWINParams(N=50, T=200, seed=12345)
        r1 = ewa_adwin_run!(params)
        r2 = ewa_adwin_run!(params)

        @test r1.tick_count == r2.tick_count
        for i in 1:r1.tick_count
            @test r1.history[i].fraction_A == r2.history[i].fraction_A
            @test r1.history[i].coordination_rate == r2.history[i].coordination_rate
            @test r1.history[i].belief_error == r2.history[i].belief_error
            @test r1.history[i].belief_variance == r2.history[i].belief_variance
        end
    end

end  # @testset "EWA-ADWIN Tests"
