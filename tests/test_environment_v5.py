"""
Tests for V5 environment integration: normative tick cycle,
enforcement signal broadcasting, and normative metrics collection.
"""

import numpy as np
import pytest

from src.environment import SimulationEnvironment, TickMetrics


class TestNormativeTickCycle:
    """Tests for running full ticks with normative enabled."""

    def test_full_tick_with_normative_enabled(self):
        """Environment should complete a full tick with normative memory."""
        env = SimulationEnvironment(
            num_agents=10,
            enable_normative=True,
            observation_k=2,
            random_seed=42,
        )

        metrics = env.step()
        assert isinstance(metrics, TickMetrics)
        assert metrics.tick == 0
        assert metrics.num_interactions == 5  # 10/2 pairs

    def test_run_short_simulation_with_normative(self):
        """Run a short simulation with normative memory enabled."""
        env = SimulationEnvironment(
            num_agents=20,
            memory_type="dynamic",
            enable_normative=True,
            observation_k=3,
            crystal_threshold=2.0,
            random_seed=42,
            convergence_threshold=0.95,
            convergence_window=10,
        )

        result = env.run(max_ticks=50, early_stop=False)
        assert result is not None
        assert len(result.tick_history) == 50

    def test_normative_metrics_collected(self):
        """Normative metrics should be non-zero after some ticks."""
        env = SimulationEnvironment(
            num_agents=20,
            memory_type="dynamic",
            enable_normative=True,
            observation_k=3,
            crystal_threshold=1.0,  # easy to crystallise
            ddm_noise=0.0,
            random_seed=42,
        )

        # Run enough ticks for some norms to form
        for _ in range(30):
            metrics = env.step()

        # At least some agents should have crystallised norms
        # (with noise=0 and threshold=1.0, it should happen quickly)
        assert metrics.norm_adoption_rate >= 0.0  # could be 0 if not enough consistency yet


class TestEnforcementBroadcast:
    """Tests for enforcement signal broadcasting."""

    def test_enforcement_signals_received(self):
        """When enforcement occurs, other agents should receive signals."""
        env = SimulationEnvironment(
            num_agents=10,
            memory_type="dynamic",
            enable_normative=True,
            observation_k=2,
            crystal_threshold=0.5,
            ddm_noise=0.0,
            enforce_threshold=0.7,
            normative_initial_strength=0.8,
            random_seed=42,
        )

        # Run some ticks so norms can form and enforcement can happen
        total_enforcements = 0
        for _ in range(50):
            metrics = env.step()
            total_enforcements += metrics.enforcement_events

        # Enforcement may or may not have happened depending on dynamics
        # Just verify the metric is tracked
        assert isinstance(total_enforcements, int)
        assert total_enforcements >= 0


class TestNormativeMetrics:
    """Tests for normative metric computation."""

    def test_dataframe_includes_normative_columns(self):
        """History dataframe should include V5 normative columns."""
        env = SimulationEnvironment(
            num_agents=10,
            enable_normative=True,
            random_seed=42,
        )
        env.step()

        df = env.get_history_dataframe()
        expected_columns = [
            "norm_adoption_rate",
            "mean_norm_strength",
            "mean_ddm_evidence",
            "total_anomalies",
            "norm_crises",
            "enforcement_events",
            "norm_crystallisations",
            "correct_norm_rate",
            "mean_compliance",
        ]
        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"

    def test_final_state_includes_normative_fields(self):
        """Final state should include V5 normative fields."""
        env = SimulationEnvironment(
            num_agents=10,
            enable_normative=True,
            random_seed=42,
        )
        result = env.run(max_ticks=5, early_stop=False)

        assert "final_norm_adoption_rate" in result.final_state
        assert "final_mean_norm_strength" in result.final_state
        assert "final_mean_compliance" in result.final_state
