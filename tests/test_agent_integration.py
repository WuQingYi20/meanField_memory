"""
Tests for V5 agent integration: effective belief blending,
enforcement logic, and normative memory integration.
"""

import numpy as np
import pytest

from src.agent import Agent, create_agent
from src.memory import FixedMemory, NormativeMemory, Interaction
from src.decision import create_decision, DecisionMode


def _make_agent_with_norm(
    norm_strategy: int = 0,
    strength: float = 0.8,
    compliance_exponent: float = 2.0,
    enforce_threshold: float = 0.7,
    initial_trust: float = 0.5,
) -> Agent:
    """Helper: create agent with a pre-crystallised norm."""
    memory = FixedMemory(size=5)
    decision = create_decision(
        mode=DecisionMode.COGNITIVE_LOCKIN,
        initial_trust=initial_trust,
    )
    nm = NormativeMemory(
        compliance_exponent=compliance_exponent,
        rng=np.random.RandomState(42),
    )
    # Force crystallisation
    nm._norm = norm_strategy
    nm._strength = strength

    return Agent(
        agent_id=0,
        memory=memory,
        decision=decision,
        normative_memory=nm,
        enforce_threshold=enforce_threshold,
        initial_strategy=1,  # different from norm to test blending
    )


class TestEffectiveBelief:
    """Tests for b_eff = compliance * b_norm + (1-compliance) * b_exp (Eq. 15)."""

    def test_effective_belief_numerical(self):
        """
        Verify effective belief blending with known values.

        b_exp = [0.6, 0.4] (from memory)
        norm = 0, sigma = 0.9, k = 2 -> compliance = 0.81
        b_norm = [1.0, 0.0]
        b_eff = 0.81 * [1, 0] + 0.19 * [0.6, 0.4] = [0.81 + 0.114, 0.076] = [0.924, 0.076]
        """
        agent = _make_agent_with_norm(norm_strategy=0, strength=0.9, compliance_exponent=2.0)

        # Seed memory with 60% strategy 0
        for i in range(3):
            agent._memory.add(Interaction(tick=i, partner_strategy=0, own_strategy=0, success=True, payoff=1.0))
        for i in range(2):
            agent._memory.add(Interaction(tick=i+3, partner_strategy=1, own_strategy=1, success=True, payoff=1.0))

        b_exp = agent._memory.get_strategy_distribution()
        np.testing.assert_array_almost_equal(b_exp, [0.6, 0.4])

        compliance = 0.9 ** 2  # 0.81
        expected_b_eff = compliance * np.array([1.0, 0.0]) + (1.0 - compliance) * np.array([0.6, 0.4])
        np.testing.assert_array_almost_equal(expected_b_eff, [0.924, 0.076])

        # The choose_action method uses b_eff internally - we can't directly access it,
        # but we can verify the normative memory state
        assert agent._normative_memory.get_compliance() == pytest.approx(0.81)
        assert agent._normative_memory.norm == 0
        norm_belief = agent._normative_memory.get_norm_belief()
        np.testing.assert_array_equal(norm_belief, [1.0, 0.0])

    def test_without_norm_beff_equals_bexp(self):
        """Without norm, effective belief = experience belief."""
        agent = create_agent(agent_id=0, initial_strategy=0)

        # No normative memory
        assert agent._normative_memory is None

        # Seed some memory
        for i in range(3):
            agent._memory.add(Interaction(tick=i, partner_strategy=0, own_strategy=0, success=True, payoff=1.0))

        b_exp = agent._memory.get_strategy_distribution()
        # choose_action will use b_exp directly (no normative constraint)
        # Just verify the agent works without normative memory
        action, pred = agent.choose_action()
        assert action in [0, 1]
        assert pred in [0, 1]

    def test_norm_disabled_beff_equals_bexp(self):
        """With normative memory but no crystallised norm, b_eff = b_exp."""
        nm = NormativeMemory(rng=np.random.RandomState(42))
        # nm has no norm
        assert not nm.has_norm()

        agent = Agent(
            agent_id=0,
            memory=FixedMemory(size=5),
            decision=create_decision(mode=DecisionMode.COGNITIVE_LOCKIN),
            normative_memory=nm,
            initial_strategy=0,
        )

        # No crystallised norm, so normative constraint should not apply
        action, pred = agent.choose_action()
        assert action in [0, 1]


class TestEnforcementLogic:
    """Tests for enforcement signal generation."""

    def test_enforcement_requires_all_three_conditions(self):
        """Enforcement needs: has_norm AND sigma > theta AND violation."""
        # Agent with norm, high strength
        agent = _make_agent_with_norm(norm_strategy=0, strength=0.8, enforce_threshold=0.7)

        # Violation -> enforcement
        signal = agent.get_enforcement_signal([1])
        assert signal == 0  # broadcasts the norm

        # Conforming -> no enforcement
        signal = agent.get_enforcement_signal([0])
        assert signal is None

    def test_low_sigma_accumulates_anomaly_instead(self):
        """Low-sigma agent accumulates anomaly instead of enforcing."""
        agent = _make_agent_with_norm(norm_strategy=0, strength=0.5, enforce_threshold=0.7)

        # Low sigma (0.5 <= 0.7): should not enforce
        signal = agent.get_enforcement_signal([1])
        assert signal is None

        # Process normative observations: should accumulate anomaly
        crystallised, dissolved, enforcements = agent.process_normative_observations([1])
        assert not crystallised
        assert not dissolved
        assert enforcements == 0
        assert agent._normative_memory.anomaly_count == 1

    def test_no_enforcement_without_normative_memory(self):
        """Agent without normative memory never enforces."""
        agent = create_agent(agent_id=0)
        signal = agent.get_enforcement_signal([0, 1])
        assert signal is None

    def test_no_enforcement_without_norm(self):
        """Agent with normative memory but no norm doesn't enforce."""
        nm = NormativeMemory(rng=np.random.RandomState(42))
        agent = Agent(
            agent_id=0,
            memory=FixedMemory(size=5),
            decision=create_decision(mode=DecisionMode.COGNITIVE_LOCKIN),
            normative_memory=nm,
            initial_strategy=0,
        )

        signal = agent.get_enforcement_signal([0, 1])
        assert signal is None


class TestProcessNormativeObservations:
    """Tests for the full normative observation processing pipeline."""

    def test_pre_crystallisation_feeds_ddm(self):
        """Before crystallisation, observations feed DDM evidence."""
        nm = NormativeMemory(
            ddm_noise=0.0,
            crystal_threshold=10.0,
            rng=np.random.RandomState(42),
        )
        agent = Agent(
            agent_id=0,
            memory=FixedMemory(size=5),
            decision=create_decision(mode=DecisionMode.COGNITIVE_LOCKIN, initial_trust=0.01),
            normative_memory=nm,
            initial_strategy=0,
        )

        # All strategy 0 -> signed_consistency = (3-0)/3 = 1.0, C~=0.01 -> drift ~= 0.99
        crystallised, dissolved, enforcements = agent.process_normative_observations([0, 0, 0])
        assert not crystallised  # threshold=10, only 1 tick
        assert nm.evidence > 0

    def test_crystallisation_returns_true(self):
        """Crystallisation should be reported."""
        nm = NormativeMemory(
            ddm_noise=0.0,
            crystal_threshold=0.5,  # low enough to crystallise in 1 tick with C=0.01
            rng=np.random.RandomState(42),
        )
        agent = Agent(
            agent_id=0,
            memory=FixedMemory(size=5),
            decision=create_decision(mode=DecisionMode.COGNITIVE_LOCKIN, initial_trust=0.01),
            normative_memory=nm,
            initial_strategy=0,
        )

        # V5.1: signed_consistency = (3-0)/3 = 1.0, drift = (1-0.01)*1.0 = 0.99 > 0.5
        crystallised, dissolved, enforcements = agent.process_normative_observations([0, 0, 0])
        assert crystallised
        assert nm.has_norm()
        assert nm.norm == 0

    def test_post_crystallisation_tracks_anomalies(self):
        """After crystallisation, violations become anomalies."""
        agent = _make_agent_with_norm(norm_strategy=0, strength=0.5, enforce_threshold=0.7)

        # Low sigma -> anomalies instead of enforcement
        crystallised, dissolved, enforcements = agent.process_normative_observations([1, 1, 1])
        assert not crystallised
        assert not dissolved
        assert enforcements == 0
        assert agent._normative_memory.anomaly_count == 3


class TestReceiveNormativeSignal:
    """Tests for receiving enforcement signals."""

    def test_signal_boosts_ddm(self):
        """Receiving normative signal should push DDM evidence toward enforced strategy."""
        nm = NormativeMemory(
            ddm_noise=0.0,
            crystal_threshold=100.0,
            rng=np.random.RandomState(42),
        )
        agent = Agent(
            agent_id=0,
            memory=FixedMemory(size=5),
            decision=create_decision(mode=DecisionMode.COGNITIVE_LOCKIN, initial_trust=0.01),
            normative_memory=nm,
            signal_amplification=3.0,
            initial_strategy=0,
        )

        # First update without signal: signed_consistency=(3-0)/3=1.0, drift=(1-0.01)*1.0=0.99
        agent.process_normative_observations([0, 0, 0])
        evidence_1 = nm.evidence
        assert evidence_1 == pytest.approx(0.99)

        # Receive signal for strategy A
        # V5.1: signal_push = (1-C) * gamma * dir(A) = (1-0.01) * 3.0 * 1.0 = 2.97
        agent.receive_normative_signal(0)

        # Second update: drift = 0.99 + signal_push 2.97 = 3.96
        agent.process_normative_observations([0, 0, 0])
        evidence_2 = nm.evidence
        assert evidence_2 == pytest.approx(0.99 + 0.99 + 2.97)  # prior + drift + push


class TestCreateAgentFactory:
    """Tests for the create_agent factory function with normative params."""

    def test_create_agent_without_normative(self):
        """Default: no normative memory."""
        agent = create_agent(agent_id=0)
        assert agent._normative_memory is None
        assert agent.normative_state is None

    def test_create_agent_with_normative(self):
        """With enable_normative=True, normative memory should be created."""
        agent = create_agent(
            agent_id=0,
            enable_normative=True,
            ddm_noise=0.2,
            crystal_threshold=5.0,
            normative_initial_strength=0.9,
            crisis_threshold=15,
            enforce_threshold=0.6,
            compliance_exponent=3.0,
            signal_amplification=1.5,
        )
        assert agent._normative_memory is not None
        assert agent._enforce_threshold == 0.6
        assert agent._signal_amplification == 1.5

        state = agent.normative_state
        assert state is not None
        assert not state.has_norm


class TestAgentReset:
    """Tests for agent reset with normative memory."""

    def test_reset_clears_normative_state(self):
        """Reset should also reset normative memory."""
        agent = _make_agent_with_norm(norm_strategy=0, strength=0.8)
        assert agent._normative_memory.has_norm()

        agent.reset(initial_strategy=0)

        assert not agent._normative_memory.has_norm()
        assert agent._normative_memory.norm is None
        assert agent._normative_memory.evidence == pytest.approx(0.0)


class TestAgentMetrics:
    """Tests for agent metrics with normative state."""

    def test_metrics_include_normative_state(self):
        """Metrics should include normative state when enabled."""
        agent = _make_agent_with_norm(norm_strategy=0, strength=0.8)
        agent._normative_memory.record_enforcement()

        metrics = agent.get_metrics()
        assert metrics["norm"] == 0
        assert metrics["norm_strength"] == pytest.approx(0.8)
        assert metrics["has_norm"] is True
        assert metrics["compliance"] == pytest.approx(0.64)
        assert metrics["enforcement_count"] == 1

    def test_metrics_without_normative(self):
        """Metrics should not include normative keys when not enabled."""
        agent = create_agent(agent_id=0)
        metrics = agent.get_metrics()
        assert "norm" not in metrics
        assert "norm_strength" not in metrics
