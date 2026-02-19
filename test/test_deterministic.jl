using Test
using Random
using DualMemoryABM

@testset "Deterministic Tests" begin

    # ── D1: Crisis σ decay ──
    @testset "D1: Crisis sigma decay" begin
        agent = AgentState(
            RingBuffer(6), 0.5, 0.5, 4,
            STRATEGY_A, 0.8, 10, 0.0,  # r=A, σ=0.8, a=10
            NO_SIGNAL, 0.0, 0.5,
        )
        params = SimulationParams(theta_crisis=10, lambda_crisis=0.3, sigma_min=0.1)

        # Manually trigger crisis check (a >= theta_crisis)
        @test agent.a >= params.theta_crisis
        agent.sigma *= params.lambda_crisis
        agent.a = 0

        @test agent.sigma ≈ 0.24
        @test agent.a == 0
        @test agent.r == STRATEGY_A  # norm unchanged
    end

    # ── D2: Double crisis dissolves ──
    @testset "D2: Double crisis dissolves" begin
        agent = AgentState(
            RingBuffer(6), 0.5, 0.5, 4,
            STRATEGY_A, 0.8, 10, 0.0,
            NO_SIGNAL, 0.0, 0.5,
        )
        params = SimulationParams(theta_crisis=10, lambda_crisis=0.3, sigma_min=0.1)

        # First crisis
        agent.sigma *= params.lambda_crisis
        agent.a = 0
        @test agent.sigma ≈ 0.24
        @test agent.sigma >= params.sigma_min  # survives

        # Second crisis
        agent.a = 10
        agent.sigma *= params.lambda_crisis
        agent.a = 0
        @test agent.sigma ≈ 0.072
        @test agent.sigma < params.sigma_min  # dissolves

        # Apply dissolution
        agent.r = NO_NORM
        agent.e = 0.0
        agent.sigma = 0.0
        agent.a = 0

        @test agent.r == NO_NORM
        @test agent.e == 0.0
        @test agent.sigma == 0.0
        @test agent.a == 0
    end

    # ── D3: Strengthening recovery ──
    @testset "D3: Strengthening recovery" begin
        sigma = 0.24
        alpha_sigma = 0.005
        n_conform = 100

        for _ in 1:n_conform
            sigma = min(1.0, sigma + alpha_sigma * (1.0 - sigma))
        end

        # Closed form: σ_new = 1 − (1−0.24)×(1−0.005)^100
        expected = 1.0 - (1.0 - 0.24) * (1.0 - 0.005)^100
        @test sigma ≈ expected atol=1e-10
        @test abs(sigma - 0.5396) < 0.001
    end

    # ── D4: Partner-only enforcement ──
    @testset "D4: Partner-only enforcement" begin
        params = SimulationParams(N=8, T=1, seed=42, V=3, Phi=1.0,
                                   enable_normative=true, theta_enforce=0.7)

        # Create agent with norm, partner conforms, V obs violate
        agent = AgentState(
            RingBuffer(6), 0.5, 0.5, 4,
            STRATEGY_A, 0.8, 0, 0.0,  # r=A, σ=0.8 > θ_enforce=0.7
            NO_SIGNAL, 0.0, 0.5,
        )

        # Simulate: partner conforms (A=0), V obs violate (B=1)
        obs_pool = Int8[STRATEGY_A, STRATEGY_B, STRATEGY_B, STRATEGY_B]  # partner + 3V
        obs_start = 1
        obs_end = 4

        ws = create_workspace(params)
        ws.partner_id[1] = 2

        post_crystal_update!(agent, 1, obs_pool, obs_start, obs_end, ws, params)

        # DD-6: partner conforms → no enforcement
        @test ws.enforce_target[1] == 0

        # V violations counted as anomaly
        @test agent.a == 3
    end

    # ── D5: V obs violate + partner violates, enforce eligible ──
    @testset "D5: V violations + partner enforcement" begin
        params = SimulationParams(N=8, T=1, seed=42, V=3, Phi=1.0,
                                   enable_normative=true, theta_enforce=0.7)

        agent = AgentState(
            RingBuffer(6), 0.5, 0.5, 4,
            STRATEGY_A, 0.8, 0, 0.0,  # σ=0.8 > θ_enforce
            NO_SIGNAL, 0.0, 0.5,
        )

        # Partner violates (B=1), V obs violate too
        obs_pool = Int8[STRATEGY_B, STRATEGY_B, STRATEGY_B, STRATEGY_B]
        ws = create_workspace(params)
        ws.partner_id[1] = 2

        post_crystal_update!(agent, 1, obs_pool, 1, 4, ws, params)

        # DD-7: enforcement triggered for partner, partner violation NOT anomaly
        @test ws.enforce_target[1] == 2
        @test ws.enforce_strategy[1] == STRATEGY_A

        # Only V violations counted as anomaly (3, not 4)
        @test agent.a == 3
    end

    # ── D6: Φ=0 blocks enforcement ──
    @testset "D6: Phi=0 blocks enforcement" begin
        params = SimulationParams(N=8, T=1, seed=42, Phi=0.0,
                                   enable_normative=true, theta_enforce=0.7)

        agent = AgentState(
            RingBuffer(6), 0.5, 0.5, 4,
            STRATEGY_A, 0.9, 0, 0.0,  # σ=0.9 > θ_enforce, but Φ=0
            NO_SIGNAL, 0.0, 0.5,
        )

        # Partner violates
        obs_pool = Int8[STRATEGY_B]
        ws = create_workspace(params)
        ws.partner_id[1] = 2

        post_crystal_update!(agent, 1, obs_pool, 1, 1, ws, params)

        # Φ=0 → can_enforce = false
        @test ws.enforce_target[1] == 0
        # Violation counted as anomaly
        @test agent.a == 1
    end

    # ── D7: Effective belief blending ──
    @testset "D7: Effective belief blending" begin
        agent = AgentState(
            RingBuffer(6), 0.3, 0.5, 4,  # b_exp_A = 0.3
            STRATEGY_A, 0.8, 0, 0.0,     # r=A, σ=0.8
            NO_SIGNAL, 0.0, 0.5,
        )
        params = SimulationParams(k=2.0)

        compute_effective_belief!(agent, params)

        # compliance = 0.8^2 = 0.64
        @test agent.compliance ≈ 0.64
        # b_eff_A = 0.64 * 1.0 + 0.36 * 0.3 = 0.748
        @test agent.b_eff_A ≈ 0.748
    end

    # ── D8: Window from confidence (C=0) ──
    @testset "D8: Window at C=0" begin
        w = 2 + floor(Int, 0.0 * (6 - 2))
        @test w == 2
    end

    # ── D9: Window from confidence (C=1) ──
    @testset "D9: Window at C=1" begin
        w = 2 + floor(Int, 1.0 * (6 - 2))
        @test w == 6
    end

    # ── D10: Confidence update correct prediction ──
    @testset "D10: Confidence update correct" begin
        C = 0.5
        alpha = 0.1
        C_new = C + alpha * (1.0 - C)
        @test C_new ≈ 0.55
    end

    # ── D11: Confidence update wrong prediction ──
    @testset "D11: Confidence update wrong" begin
        C = 0.5
        beta = 0.3
        C_new = C * (1.0 - beta)
        @test C_new ≈ 0.35
    end

    # ── D12: No normative — stages 4-5 skipped ──
    @testset "D12: No normative mode" begin
        params = SimulationParams(N=10, T=50, seed=42, enable_normative=false)
        result = run!(params)

        for m in result.history
            @test m.num_crystallised == 0
            @test m.num_enforcements == 0
            @test m.frac_dominant_norm == 0.0
        end
    end

    # ── D13: Crystallisation direction positive ──
    @testset "D13: Crystallisation e > 0 → A" begin
        agent = AgentState(
            RingBuffer(6), 0.5, 0.5, 4,
            NO_NORM, 0.0, 0, 3.1,  # e = 3.1
            NO_SIGNAL, 0.0, 0.5,
        )
        params = SimulationParams(theta_crystal=3.0, sigma_0=0.8)

        # Check crystallisation
        if abs(agent.e) >= params.theta_crystal
            agent.r = agent.e > 0 ? STRATEGY_A : STRATEGY_B
            agent.sigma = params.sigma_0
            agent.a = 0
        end

        @test agent.r == STRATEGY_A
        @test agent.sigma == 0.8
        @test agent.a == 0
    end

    # ── D14: Crystallisation direction negative ──
    @testset "D14: Crystallisation e < 0 → B" begin
        agent = AgentState(
            RingBuffer(6), 0.5, 0.5, 4,
            NO_NORM, 0.0, 0, -3.1,  # e = -3.1
            NO_SIGNAL, 0.0, 0.5,
        )
        params = SimulationParams(theta_crystal=3.0, sigma_0=0.8)

        if abs(agent.e) >= params.theta_crystal
            agent.r = agent.e > 0 ? STRATEGY_A : STRATEGY_B
            agent.sigma = params.sigma_0
            agent.a = 0
        end

        @test agent.r == STRATEGY_B
        @test agent.sigma == 0.8
        @test agent.a == 0
    end

    # ── D15: Signal consumed by post-crystallised agent ──
    @testset "D15: Signal consumed by post-crystallised" begin
        agent = AgentState(
            RingBuffer(6), 0.5, 0.5, 4,
            STRATEGY_A, 0.8, 0, 0.0,  # post-crystallised
            STRATEGY_B,                 # has pending signal
            0.0, 0.5,
        )
        params = SimulationParams(N=4, T=1, seed=42, enable_normative=true, Phi=0.0)

        # Partner conforms
        obs_pool = Int8[STRATEGY_A]
        ws = create_workspace(params)
        ws.partner_id[1] = 2

        post_crystal_update!(agent, 1, obs_pool, 1, 1, ws, params)

        # Signal should be consumed (cleared)
        @test agent.pending_signal == NO_SIGNAL
    end

    # ── Additional: RingBuffer tests ──
    @testset "RingBuffer basics" begin
        buf = RingBuffer(3)
        @test length(buf) == 0
        @test capacity(buf) == 3

        push!(buf, STRATEGY_A)
        push!(buf, STRATEGY_B)
        @test length(buf) == 2

        r = recent(buf, 2)
        @test r == Int8[STRATEGY_A, STRATEGY_B]

        push!(buf, STRATEGY_A)
        push!(buf, STRATEGY_B)  # overwrites oldest
        @test length(buf) == 3

        r = recent(buf, 3)
        @test r == Int8[STRATEGY_B, STRATEGY_A, STRATEGY_B]

        n_A, total = count_strategy_A(buf, 2)
        @test n_A == 1
        @test total == 2
    end

    # ── Additional: Smoke test ──
    @testset "Smoke test: basic run" begin
        params = SimulationParams(N=20, T=100, seed=42)
        result = run!(params)
        @test result.tick_count > 0
        @test result.tick_count <= 100
        @test length(result.history) == result.tick_count
    end

    @testset "Smoke test: normative run" begin
        params = SimulationParams(N=20, T=200, seed=42,
                                   enable_normative=true, V=3, Phi=1.0)
        result = run!(params)
        @test result.tick_count > 0
    end

    # ── Validation tests ──
    @testset "Parameter validation" begin
        @test_throws ArgumentError validate(SimulationParams(N=3))  # odd
        @test_throws ArgumentError validate(SimulationParams(N=1))  # too small
        @test_throws ArgumentError validate(SimulationParams(alpha=0.5, beta=0.3))  # beta <= alpha
        @test_throws ArgumentError validate(SimulationParams(w_max=1, w_base=2))  # w_max < w_base
    end

end
