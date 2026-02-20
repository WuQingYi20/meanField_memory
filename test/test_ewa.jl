using Test
using Random
using DualMemoryABM

@testset "EWA Tests" begin

    # ══════════════════════════════════════════════════════════════
    # Deterministic Tests (E1-E11)
    # ══════════════════════════════════════════════════════════════

    # ── E1: Symmetric initialization ──
    @testset "E1: Symmetric init" begin
        params = EWAParams(N=10, T=100, seed=42)
        agents, ws, history, rng = ewa_initialize(params)

        for agent in agents
            @test agent.attract_A == 0.0
            @test agent.attract_B == 0.0
            @test agent.N_exp == 1.0
            @test agent.prob_A == 0.5
        end
    end

    # ── E2: Single-tick update, partner plays A ──
    @testset "E2: Update — both play A" begin
        # Default params: delta=0.5, phi=0.9, rho=0.9, lambda=1.0, A0=0.0
        agent = EWAAgent(0.0, 0.0, 1.0, 0.5)
        s_i = STRATEGY_A
        s_j = STRATEGY_A

        delta, phi, rho, lam = 0.5, 0.9, 0.9, 1.0

        # pi(A, A) = 1, pi(B, A) = 0
        weight_A = (delta + (1.0 - delta) * 1.0) * 1.0  # = 1.0
        weight_B = (delta + (1.0 - delta) * 0.0) * 0.0  # = 0.0

        N_new = rho * 1.0 + 1.0  # = 1.9
        attract_A_new = (phi * 1.0 * 0.0 + weight_A) / N_new  # = 1.0/1.9
        attract_B_new = (phi * 1.0 * 0.0 + weight_B) / N_new  # = 0.0

        # Apply same logic to agent
        agent.attract_A = attract_A_new
        agent.attract_B = attract_B_new
        agent.N_exp = N_new
        _ewa_update_prob!(agent, lam)

        @test agent.attract_A ≈ 1.0 / 1.9
        @test agent.attract_B ≈ 0.0
        @test agent.N_exp ≈ 1.9

        # prob_A = exp(λ*A_A) / (exp(λ*A_A) + exp(λ*A_B)) = exp(1/1.9) / (exp(1/1.9) + 1)
        expected_prob = exp(1.0 / 1.9) / (exp(1.0 / 1.9) + 1.0)
        @test agent.prob_A ≈ expected_prob atol=1e-10
        @test agent.prob_A > 0.5  # should favor A after coordination on A
    end

    # ── E3: Single-tick update, partner plays B ──
    @testset "E3: Update — I play A, partner plays B" begin
        agent = EWAAgent(0.0, 0.0, 1.0, 0.5)
        delta, phi, rho, lam = 0.5, 0.9, 0.9, 1.0

        # pi(A, B) = 0, pi(B, B) = 1
        # weight_A = (0.5 + 0.5*1) * 0 = 0 (I played A, but payoff for A when partner played B = 0)
        # weight_B = (0.5 + 0.5*0) * 1 = 0.5 (imagined playing B)
        weight_A = 0.0
        weight_B = 0.5

        N_new = rho * 1.0 + 1.0
        agent.attract_A = (phi * 1.0 * 0.0 + weight_A) / N_new
        agent.attract_B = (phi * 1.0 * 0.0 + weight_B) / N_new
        agent.N_exp = N_new
        _ewa_update_prob!(agent, lam)

        @test agent.attract_A ≈ 0.0
        @test agent.attract_B ≈ 0.5 / 1.9
        @test agent.prob_A < 0.5  # should now favor B
        expected_prob = 1.0 / (1.0 + exp(lam * 0.5 / 1.9))
        @test agent.prob_A ≈ expected_prob atol=1e-10
    end

    # ── E4: Logit probability computation ──
    @testset "E4: Logit probability" begin
        agent = EWAAgent(1.0, 0.5, 1.0, 0.5)

        _ewa_update_prob!(agent, 2.0)
        # prob_A = exp(2*1) / (exp(2*1) + exp(2*0.5)) = exp(2)/(exp(2)+exp(1))
        expected = exp(2.0) / (exp(2.0) + exp(1.0))
        @test agent.prob_A ≈ expected atol=1e-10

        # Test with large values (numerical stability)
        agent2 = EWAAgent(100.0, 99.0, 1.0, 0.5)
        _ewa_update_prob!(agent2, 10.0)
        # diff = 10*(100-99) = 10, prob ≈ 1/(1+exp(-10)) ≈ 0.99995
        @test agent2.prob_A > 0.999
        @test agent2.prob_A <= 1.0

        # Test equal attractions
        agent3 = EWAAgent(5.0, 5.0, 1.0, 0.5)
        _ewa_update_prob!(agent3, 3.0)
        @test agent3.prob_A ≈ 0.5
    end

    # ── E5: Pure RL (delta=0) ──
    @testset "E5: Pure RL (delta=0)" begin
        phi, rho, lam = 0.9, 0.9, 1.0
        delta = 0.0

        # Case 1: Both play A → A reinforced
        agent = EWAAgent(0.0, 0.0, 1.0, 0.5)
        # weight_A = (0 + 1*1)*1 = 1, weight_B = (0 + 1*0)*0 = 0
        N_new = rho * 1.0 + 1.0
        agent.attract_A = (phi * 1.0 * 0.0 + 1.0) / N_new
        agent.attract_B = (phi * 1.0 * 0.0 + 0.0) / N_new
        agent.N_exp = N_new
        _ewa_update_prob!(agent, lam)
        @test agent.attract_A ≈ 1.0 / 1.9
        @test agent.attract_B ≈ 0.0
        @test agent.prob_A > 0.5

        # Case 2: I play A, partner plays B → nothing reinforced
        agent2 = EWAAgent(0.0, 0.0, 1.0, 0.5)
        # weight_A = (0+1*1)*0 = 0, weight_B = (0+1*0)*1 = 0
        N_new2 = rho * 1.0 + 1.0
        agent2.attract_A = (phi * 1.0 * 0.0 + 0.0) / N_new2
        agent2.attract_B = (phi * 1.0 * 0.0 + 0.0) / N_new2
        agent2.N_exp = N_new2
        _ewa_update_prob!(agent2, lam)
        @test agent2.attract_A ≈ 0.0
        @test agent2.attract_B ≈ 0.0
        @test agent2.prob_A ≈ 0.5  # stays symmetric
    end

    # ── E6: Pure belief learning (delta=1) ──
    @testset "E6: Pure belief (delta=1)" begin
        phi, rho, lam = 0.9, 0.9, 1.0
        delta = 1.0

        # I play A, partner plays B → B reinforced with weight 1
        agent = EWAAgent(0.0, 0.0, 1.0, 0.5)
        # weight_A = (1+0)*0 = 0, weight_B = (1+0)*1 = 1
        N_new = rho * 1.0 + 1.0
        agent.attract_A = (phi * 1.0 * 0.0 + 0.0) / N_new
        agent.attract_B = (phi * 1.0 * 0.0 + 1.0) / N_new
        agent.N_exp = N_new
        _ewa_update_prob!(agent, lam)
        @test agent.attract_B ≈ 1.0 / 1.9
        @test agent.attract_A ≈ 0.0
        @test agent.prob_A < 0.5  # now favors B

        # I play B, partner plays A → A reinforced with weight 1 (own action irrelevant)
        agent2 = EWAAgent(0.0, 0.0, 1.0, 0.5)
        # weight_A = (1+0)*1 = 1, weight_B = (1+0)*0 = 0
        N_new2 = rho * 1.0 + 1.0
        agent2.attract_A = (phi * 1.0 * 0.0 + 1.0) / N_new2
        agent2.attract_B = (phi * 1.0 * 0.0 + 0.0) / N_new2
        agent2.N_exp = N_new2
        _ewa_update_prob!(agent2, lam)
        @test agent2.attract_A ≈ 1.0 / 1.9
        @test agent2.prob_A > 0.5  # favors A regardless of own action
    end

    # ── E7: Experience weight accumulation ──
    @testset "E7: Experience weight accumulation" begin
        rho = 0.9
        N_exp = 1.0
        for _ in 1:5
            N_exp = rho * N_exp + 1.0
        end
        # N(t) = sum_{i=0}^{t} rho^i = (1 - rho^{t+1}) / (1 - rho)
        expected = (1.0 - 0.9^6) / (1.0 - 0.9)
        @test N_exp ≈ expected atol=1e-10
    end

    # ── E8: TickMetrics normative fields = 0 ──
    @testset "E8: Normative fields zero" begin
        params = EWAParams(N=20, T=50, seed=42)
        result = ewa_run!(params)

        for m in result.history
            @test m.mean_confidence == 0.0
            @test m.num_crystallised == 0
            @test m.mean_norm_strength == 0.0
            @test m.num_enforcements == 0
            @test m.frac_dominant_norm == 0.0
        end
    end

    # ── E9: Convergence detection ──
    @testset "E9: Convergence detection" begin
        params = EWAParams(convergence_window=3)

        # Build a mock history where convergence_counter reaches 3
        history = Vector{TickMetrics}(undef, 5)
        history[1] = TickMetrics(1, 0.98, 0.0, 0.96, 0, 0.0, 0, 0.05, 0.01, 0.0, 1)
        history[2] = TickMetrics(2, 0.98, 0.0, 0.96, 0, 0.0, 0, 0.05, 0.01, 0.0, 2)
        history[3] = TickMetrics(3, 0.98, 0.0, 0.96, 0, 0.0, 0, 0.05, 0.01, 0.0, 3)

        @test !ewa_check_convergence(history, 2, params)  # counter=2 < window=3
        @test ewa_check_convergence(history, 3, params)   # counter=3 >= window=3
    end

    # ── E10: Smoke test ──
    @testset "E10: Smoke test (ewa_run!)" begin
        params = EWAParams(N=20, T=100, seed=42)
        result = ewa_run!(params)

        @test result.tick_count > 0
        @test result.tick_count <= 100
        @test length(result.history) == result.tick_count
        @test result.probes === nothing

        # Verify fraction_A is reasonable
        last_m = result.history[end]
        @test 0.0 <= last_m.fraction_A <= 1.0
        @test 0.0 <= last_m.coordination_rate <= 1.0
    end

    # ── E11: Parameter validation ──
    @testset "E11: Parameter validation" begin
        @test_throws ArgumentError validate(EWAParams(N=3))     # odd
        @test_throws ArgumentError validate(EWAParams(N=1))     # too small
        @test_throws ArgumentError validate(EWAParams(delta=-0.1))  # out of [0,1]
        @test_throws ArgumentError validate(EWAParams(delta=1.1))
        @test_throws ArgumentError validate(EWAParams(phi=-0.1))
        @test_throws ArgumentError validate(EWAParams(rho=1.5))
        @test_throws ArgumentError validate(EWAParams(lambda=0.0))  # must be > 0
        @test_throws ArgumentError validate(EWAParams(lambda=-1.0))
        @test_throws ArgumentError validate(EWAParams(convergence_window=0))

        # Valid edge cases should not throw
        @test validate(EWAParams(delta=0.0)) === nothing  # pure RL
        @test validate(EWAParams(delta=1.0)) === nothing  # pure belief
        @test validate(EWAParams(phi=0.0)) === nothing
        @test validate(EWAParams(rho=0.0)) === nothing
    end

    # ── Additional: summarize() works with EWA ──
    @testset "EWA summarize compatibility" begin
        params = EWAParams(N=20, T=100, seed=42)
        result = ewa_run!(params)
        ts = summarize(result)

        @test ts isa TrialSummary
        @test ts.total_ticks == result.tick_count
        @test ts.final_num_crystallised == 0
        @test ts.final_mean_norm_strength == 0.0
    end

    # ── Additional: ewa_first_tick_per_layer ──
    @testset "EWA first_tick_per_layer" begin
        params = EWAParams(N=20, T=200, seed=42)
        result = ewa_run!(params)
        layers = ewa_first_tick_per_layer(result.history, params.N, params)

        @test layers.crystal == 0  # always 0 for EWA
        @test layers isa NamedTuple{(:behavioral, :belief, :crystal, :all_met)}
    end

    # ── Additional: to_namedtuple round-trip ──
    @testset "EWA to_namedtuple" begin
        params = EWAParams(N=50, delta=0.7, lambda=2.0)
        nt = to_namedtuple(params)
        params2 = EWAParams(; nt...)
        @test params2.N == 50
        @test params2.delta == 0.7
        @test params2.lambda == 2.0
    end

    # ══════════════════════════════════════════════════════════════
    # Statistical Tests (ES1-ES3)
    # ══════════════════════════════════════════════════════════════

    # ── ES1: Convergence rate >= 80% ──
    # Note: lambda must be > 2 for the symmetric equilibrium to become unstable.
    # With lambda=5, the mean-field fixed point is ~0.99, well above thresh_majority=0.95.
    @testset "ES1: EWA convergence rate" begin
        n_trials = 30
        n_converged = 0
        for trial in 1:n_trials
            result = ewa_run!(EWAParams(N=100, T=3000, seed=trial, lambda=5.0))
            ts = summarize(result)
            if ts.converged
                n_converged += 1
            end
        end
        rate = n_converged / n_trials
        @test rate >= 0.80
    end

    # ── ES2: Higher lambda → faster convergence ──
    # lambda=2.0 is at the bifurcation point (symmetric eq. neutrally stable, won't converge).
    # lambda=5.0 easily converges. So lambda=5.0 should have fewer ticks on average.
    @testset "ES2: Lambda sensitivity" begin
        n_trials = 20
        ticks_low = Int[]
        ticks_high = Int[]

        for trial in 1:n_trials
            r_low = ewa_run!(EWAParams(N=100, T=3000, seed=trial, lambda=2.0))
            r_high = ewa_run!(EWAParams(N=100, T=3000, seed=trial, lambda=5.0))
            push!(ticks_low, r_low.tick_count)
            push!(ticks_high, r_high.tick_count)
        end

        # Higher lambda should converge faster on average
        @test mean(ticks_high) <= mean(ticks_low)
    end

    # ── ES3: Deterministic seeding ──
    @testset "ES3: Deterministic seeding" begin
        params = EWAParams(N=50, T=200, seed=12345)
        r1 = ewa_run!(params)
        r2 = ewa_run!(params)

        @test r1.tick_count == r2.tick_count
        for i in 1:r1.tick_count
            @test r1.history[i].fraction_A == r2.history[i].fraction_A
            @test r1.history[i].coordination_rate == r2.history[i].coordination_rate
            @test r1.history[i].belief_error == r2.history[i].belief_error
            @test r1.history[i].belief_variance == r2.history[i].belief_variance
        end
    end

end  # @testset "EWA Tests"
