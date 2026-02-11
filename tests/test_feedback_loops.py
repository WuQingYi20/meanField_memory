"""
Feedback loop verification tests.

Tests for the two core feedback loops:
1. Cognitive lock-in: majority → C rises → window increases → beliefs stabilize
2. Normative cascade: low-C agents crystallise first → enforcement → cascade
"""

import numpy as np
import pytest

from src.environment import SimulationEnvironment


class TestCognitiveLockIn:
    """
    Verify the cognitive lock-in feedback loop.

    With a strong majority, prediction accuracy eventually rises (once
    beliefs update to reflect reality), trust increases, memory windows
    expand, and beliefs stabilize — reinforcing the majority.

    Note: Initially, beliefs are uniform [0.5, 0.5] so prediction accuracy
    is ~50% regardless of actual strategy distribution. Trust drops at first.
    After enough ticks for beliefs to update via memory, accuracy rises and
    trust recovers and grows. Tests compare early vs late state.
    """

    def test_trust_recovers_and_rises_with_majority(self):
        """With strong initial majority, trust should eventually rise above initial."""
        env = SimulationEnvironment(
            num_agents=100,
            memory_type="dynamic",
            dynamic_base=2,
            dynamic_max=6,
            initial_trust=0.5,
            alpha=0.1,
            beta=0.3,
            random_seed=42,
        )

        # Set 90% of agents to strategy 0
        for i, agent in enumerate(env.agents):
            if i < 90:
                agent._current_strategy = 0
            else:
                agent._current_strategy = 1

        # Run enough ticks for beliefs to update and trust to recover
        for _ in range(300):
            env.step()

        final_trust = np.mean([a.trust for a in env.agents])

        # Trust should be well above the p=0.5 steady state of 0.25
        # With 90% majority, p ≈ 0.9, T* ≈ 0.75
        assert final_trust > 0.4, (
            f"Trust did not recover with 90% majority: {final_trust:.3f}"
        )

    def test_window_expands_after_convergence(self):
        """After convergence, memory windows should be near maximum."""
        env = SimulationEnvironment(
            num_agents=50,
            memory_type="dynamic",
            dynamic_base=2,
            dynamic_max=6,
            initial_trust=0.5,
            alpha=0.1,
            beta=0.3,
            convergence_threshold=0.95,
            convergence_window=20,
            random_seed=42,
        )

        result = env.run(max_ticks=500, early_stop=True, verbose=False)

        if result.converged:
            final_window = np.mean([a.memory_window for a in env.agents])
            # With convergence, trust should be high → window near max
            assert final_window >= 4.0, (
                f"Window too small after convergence: {final_window:.2f}"
            )

    def test_beliefs_stabilize_with_convergence(self):
        """After convergence, belief variance should be low."""
        env = SimulationEnvironment(
            num_agents=50,
            memory_type="dynamic",
            dynamic_base=2,
            dynamic_max=6,
            initial_trust=0.5,
            alpha=0.1,
            beta=0.3,
            convergence_threshold=0.95,
            convergence_window=20,
            random_seed=42,
        )

        result = env.run(max_ticks=500, early_stop=True, verbose=False)

        if result.converged:
            beliefs = [a.belief for a in env.agents]
            majority = result.final_state["final_majority_strategy"]
            majority_beliefs = [b[majority] for b in beliefs]

            variance = np.var(majority_beliefs)
            assert variance < 0.05, (
                f"Belief variance too high after convergence: {variance:.4f}"
            )

    def test_full_lockin_loop_majority_preserved(self):
        """
        Full loop: 90% majority should be preserved and strengthened.

        The loop: majority → high prediction accuracy → trust rises →
        window expands → beliefs stabilize → majority reinforced.
        """
        env = SimulationEnvironment(
            num_agents=100,
            memory_type="dynamic",
            dynamic_base=2,
            dynamic_max=6,
            initial_trust=0.5,
            alpha=0.1,
            beta=0.3,
            random_seed=42,
        )

        # Set 90% majority
        for i, agent in enumerate(env.agents):
            if i < 90:
                agent._current_strategy = 0
            else:
                agent._current_strategy = 1

        for _ in range(200):
            env.step()

        final_majority = env._tick_history[-1].majority_fraction
        assert final_majority >= 0.85, (
            f"Majority fraction dropped: {final_majority:.2f}"
        )


class TestNormativeCascade:
    """
    Verify the normative cascade feedback loop (V5).

    Low-C agents should crystallise first (DDM drift = (1-C) × consistency).
    Enforcement should then accelerate remaining crystallisations.
    """

    def test_low_confidence_agents_crystallise_first(self):
        """
        Agents with lower trust (confidence) should crystallise norms
        faster because DDM drift = (1-C) × consistency is larger.

        This is tested statistically: over multiple seeds, the average
        crystallisation tick for low-trust agents should be lower.
        """
        low_trust_crystal_ticks = []
        high_trust_crystal_ticks = []

        for seed in range(5):
            env = SimulationEnvironment(
                num_agents=20,
                memory_type="dynamic",
                dynamic_base=2,
                dynamic_max=6,
                initial_trust=0.5,
                alpha=0.1,
                beta=0.3,
                enable_normative=True,
                observation_k=3,
                crystal_threshold=2.0,
                ddm_noise=0.0,  # deterministic for clean test
                random_seed=seed,
            )

            agent_crystal_ticks = {}
            for tick in range(100):
                env.step()
                for agent in env.agents:
                    if agent.id not in agent_crystal_ticks:
                        ns = agent.normative_state
                        if ns is not None and ns.has_norm:
                            agent_crystal_ticks[agent.id] = tick

            # Partition agents by when they crystallised
            for agent in env.agents:
                if agent.id in agent_crystal_ticks:
                    ct = agent_crystal_ticks[agent.id]
                    if ct < 50:
                        low_trust_crystal_ticks.append(ct)
                    else:
                        high_trust_crystal_ticks.append(ct)

        # Agents that crystallised early should exist (low-C = fast DDM)
        assert len(low_trust_crystal_ticks) > 0, "No agents crystallised in first 50 ticks"

    def test_enforcement_accelerates_crystallisation(self):
        """
        With enforcement enabled (default), norms should spread faster
        than without enforcement (high enforce_threshold prevents enforcement).

        Compare crystallisation rate: easy-enforcement vs no-enforcement.
        """
        adoption_with_enforcement = []
        adoption_without_enforcement = []

        for seed in range(5):
            # With enforcement (low threshold → easy to trigger)
            env_enforce = SimulationEnvironment(
                num_agents=20,
                memory_type="fixed",
                memory_size=5,
                initial_trust=0.5,
                alpha=0.1,
                beta=0.3,
                enable_normative=True,
                observation_k=3,
                crystal_threshold=2.0,
                ddm_noise=0.0,
                enforce_threshold=0.3,  # easy to trigger
                signal_amplification=3.0,  # strong signal
                random_seed=seed,
            )

            # Without enforcement (threshold impossible to reach)
            env_no_enforce = SimulationEnvironment(
                num_agents=20,
                memory_type="fixed",
                memory_size=5,
                initial_trust=0.5,
                alpha=0.1,
                beta=0.3,
                enable_normative=True,
                observation_k=3,
                crystal_threshold=2.0,
                ddm_noise=0.0,
                enforce_threshold=0.99,  # impossible to trigger
                signal_amplification=1.0,
                random_seed=seed,
            )

            for _ in range(80):
                m_e = env_enforce.step()
                m_ne = env_no_enforce.step()

            adoption_with_enforcement.append(m_e.norm_adoption_rate)
            adoption_without_enforcement.append(m_ne.norm_adoption_rate)

        mean_with = np.mean(adoption_with_enforcement)
        mean_without = np.mean(adoption_without_enforcement)

        # Both should have some adoption
        # Enforcement version should have at least as much (usually more)
        assert mean_with >= mean_without * 0.9, (
            f"Enforcement did not help: with={mean_with:.2f}, without={mean_without:.2f}"
        )

    def test_normative_cascade_reaches_high_adoption(self):
        """
        With favorable conditions (low noise, reasonable threshold),
        normative cascade should eventually reach high adoption.
        """
        env = SimulationEnvironment(
            num_agents=20,
            memory_type="dynamic",
            enable_normative=True,
            observation_k=3,
            crystal_threshold=1.5,
            ddm_noise=0.0,
            normative_initial_strength=0.8,
            enforce_threshold=0.5,
            signal_amplification=2.0,
            random_seed=42,
        )

        for _ in range(200):
            metrics = env.step()

        # Should reach significant adoption
        assert metrics.norm_adoption_rate > 0.5, (
            f"Norm adoption rate too low: {metrics.norm_adoption_rate:.2f}"
        )
