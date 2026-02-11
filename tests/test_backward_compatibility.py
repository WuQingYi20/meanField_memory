"""
Tests for backward compatibility: V1 experience-only mode must be
unchanged when normative memory is disabled.
"""

import numpy as np
import pytest

from src.environment import SimulationEnvironment


class TestBackwardCompatibility:
    """Tests that normative=False produces identical V1 behaviour."""

    def test_normative_disabled_all_metrics_zero(self):
        """With enable_normative=False, all normative metrics should be 0."""
        env = SimulationEnvironment(
            num_agents=10,
            enable_normative=False,
            random_seed=42,
        )

        for _ in range(20):
            metrics = env.step()

        assert metrics.norm_adoption_rate == 0.0
        assert metrics.mean_norm_strength == 0.0
        assert metrics.mean_ddm_evidence == 0.0
        assert metrics.total_anomalies == 0
        assert metrics.norm_crises == 0
        assert metrics.enforcement_events == 0
        assert metrics.norm_crystallisations == 0
        assert metrics.mean_compliance == 0.0

    def test_normative_disabled_no_norms_form(self):
        """With normative disabled, no agent should have a norm."""
        env = SimulationEnvironment(
            num_agents=20,
            enable_normative=False,
            random_seed=42,
        )

        env.run(max_ticks=100, early_stop=False)

        for agent in env.agents:
            assert agent.normative_state is None

    def test_same_seed_same_results_v1_mode(self):
        """Same seed should produce identical results in V1 mode."""
        def run_v1(seed):
            env = SimulationEnvironment(
                num_agents=10,
                memory_type="fixed",
                enable_normative=False,
                random_seed=seed,
            )
            result = env.run(max_ticks=50, early_stop=False)
            return result.final_state

        result1 = run_v1(42)
        result2 = run_v1(42)

        assert result1["final_majority_fraction"] == result2["final_majority_fraction"]
        assert result1["final_mean_trust"] == result2["final_mean_trust"]

    def test_agent_count_conserved(self):
        """All agents should be present after simulation."""
        env = SimulationEnvironment(
            num_agents=20,
            enable_normative=True,
            random_seed=42,
        )

        env.run(max_ticks=50, early_stop=False)
        assert len(env.agents) == 20

    def test_all_strategies_valid(self):
        """All agent strategies should be in {0, 1}."""
        env = SimulationEnvironment(
            num_agents=20,
            enable_normative=True,
            random_seed=42,
        )

        env.run(max_ticks=50, early_stop=False)

        for agent in env.agents:
            assert agent.current_strategy in [0, 1]
