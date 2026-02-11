"""
Tests for NormativeMemory: DDM crystallisation, anomaly tracking,
crisis/dissolution, compliance, enforcement, and signal reception.
"""

import numpy as np
import pytest

from src.memory.normative import NormativeMemory, NormativeState


class TestDDMCrystallisation:
    """Tests for drift-diffusion norm formation (Eq. 5-7)."""

    def test_deterministic_crystallisation_at_predicted_tick(self):
        """With noise=0, consistency=1, C=0: drift=+1/tick. Should crystallise at tick 3 for theta=3."""
        nm = NormativeMemory(
            ddm_noise=0.0,
            crystal_threshold=3.0,
            rng=np.random.RandomState(42),
        )

        # Tick 1: evidence goes from 0 to 1
        result = nm.update_evidence(confidence=0.0, consistency=1.0, dominant_strategy=0)
        assert not result
        assert nm.evidence == pytest.approx(1.0)

        # Tick 2: evidence goes from 1 to 2
        result = nm.update_evidence(confidence=0.0, consistency=1.0, dominant_strategy=0)
        assert not result
        assert nm.evidence == pytest.approx(2.0)

        # Tick 3: evidence goes from 2 to 3 -> crystallise
        result = nm.update_evidence(confidence=0.0, consistency=1.0, dominant_strategy=0)
        assert result
        assert nm.has_norm()
        assert nm.norm == 0

    def test_no_crystallisation_when_confidence_is_one(self):
        """When C=1, drift=0. Evidence should never accumulate."""
        nm = NormativeMemory(
            ddm_noise=0.0,
            crystal_threshold=3.0,
            rng=np.random.RandomState(42),
        )

        for _ in range(100):
            result = nm.update_evidence(confidence=1.0, consistency=1.0, dominant_strategy=0)
            assert not result

        assert nm.evidence == pytest.approx(0.0)
        assert not nm.has_norm()

    def test_low_confidence_crystallises_faster_than_high(self):
        """Lower C -> higher drift -> faster crystallisation."""
        nm_low = NormativeMemory(
            ddm_noise=0.0, crystal_threshold=3.0,
            rng=np.random.RandomState(42),
        )
        nm_high = NormativeMemory(
            ddm_noise=0.0, crystal_threshold=3.0,
            rng=np.random.RandomState(42),
        )

        ticks_low = None
        ticks_high = None

        for t in range(1, 1000):
            if ticks_low is None:
                if nm_low.update_evidence(confidence=0.2, consistency=1.0, dominant_strategy=0):
                    ticks_low = t
            if ticks_high is None:
                if nm_high.update_evidence(confidence=0.7, consistency=1.0, dominant_strategy=0):
                    ticks_high = t
            if ticks_low is not None and ticks_high is not None:
                break

        assert ticks_low is not None
        assert ticks_high is not None
        assert ticks_low < ticks_high

    def test_crystallisation_sets_correct_state(self):
        """On crystallisation, norm, strength, and anomaly count are set correctly."""
        nm = NormativeMemory(
            ddm_noise=0.0,
            crystal_threshold=2.0,
            initial_strength=0.8,
            rng=np.random.RandomState(42),
        )

        nm.update_evidence(confidence=0.0, consistency=1.0, dominant_strategy=1)
        nm.update_evidence(confidence=0.0, consistency=1.0, dominant_strategy=1)

        assert nm.has_norm()
        assert nm.norm == 1
        assert nm.strength == pytest.approx(0.8)
        assert nm.anomaly_count == 0

    def test_update_evidence_noop_after_crystallisation(self):
        """Once norm exists, update_evidence should return False and not change state."""
        nm = NormativeMemory(
            ddm_noise=0.0,
            crystal_threshold=1.0,
            rng=np.random.RandomState(42),
        )

        nm.update_evidence(confidence=0.0, consistency=1.0, dominant_strategy=0)
        assert nm.has_norm()

        strength_before = nm.strength
        result = nm.update_evidence(confidence=0.0, consistency=1.0, dominant_strategy=1)
        assert not result
        assert nm.norm == 0  # unchanged
        assert nm.strength == pytest.approx(strength_before)

    def test_evidence_floors_at_zero(self):
        """Evidence should not go below 0 (max(0, ...) in Eq. 6)."""
        nm = NormativeMemory(
            ddm_noise=0.0,
            crystal_threshold=3.0,
            rng=np.random.RandomState(42),
        )

        # confidence=1 -> drift=0, no noise -> stays at 0
        nm.update_evidence(confidence=1.0, consistency=1.0, dominant_strategy=0)
        assert nm.evidence >= 0.0


class TestAnomalyTracking:
    """Tests for anomaly accumulation (Eq. 10)."""

    def test_exact_anomaly_counting(self):
        """7 violations = 7 anomalies."""
        nm = NormativeMemory(
            ddm_noise=0.0,
            crystal_threshold=1.0,
            crisis_threshold=100,  # high so no crisis during test
            rng=np.random.RandomState(42),
        )

        # Crystallise a norm for strategy 0
        nm.update_evidence(confidence=0.0, consistency=1.0, dominant_strategy=0)
        assert nm.has_norm()
        assert nm.norm == 0

        # 7 violations (observing strategy 1 when norm is 0)
        for _ in range(7):
            nm.record_anomaly(observed_strategy=1)

        assert nm.anomaly_count == 7

    def test_conforming_observations_dont_change_anomalies(self):
        """Observations matching the norm should NOT increment anomaly count."""
        nm = NormativeMemory(
            ddm_noise=0.0,
            crystal_threshold=1.0,
            rng=np.random.RandomState(42),
        )

        nm.update_evidence(confidence=0.0, consistency=1.0, dominant_strategy=0)
        assert nm.has_norm()

        for _ in range(50):
            nm.record_anomaly(observed_strategy=0)  # conforming

        assert nm.anomaly_count == 0

    def test_anomaly_noop_without_norm(self):
        """Without a norm, record_anomaly should do nothing."""
        nm = NormativeMemory(rng=np.random.RandomState(42))
        assert not nm.has_norm()

        nm.record_anomaly(observed_strategy=0)
        nm.record_anomaly(observed_strategy=1)
        assert nm.anomaly_count == 0

    def test_mixed_observations(self):
        """Mix of conforming and violating observations."""
        nm = NormativeMemory(
            ddm_noise=0.0,
            crystal_threshold=1.0,
            crisis_threshold=100,
            rng=np.random.RandomState(42),
        )

        nm.update_evidence(confidence=0.0, consistency=1.0, dominant_strategy=0)

        # 3 conforming, 5 violating
        for _ in range(3):
            nm.record_anomaly(observed_strategy=0)
        for _ in range(5):
            nm.record_anomaly(observed_strategy=1)

        assert nm.anomaly_count == 5


class TestCrisisAndDissolution:
    """Tests for crisis triggering and norm dissolution (Eq. 11-12)."""

    def test_crisis_at_threshold(self):
        """10 anomalies -> strength *= 0.3, anomalies reset."""
        nm = NormativeMemory(
            ddm_noise=0.0,
            crystal_threshold=1.0,
            initial_strength=0.8,
            crisis_threshold=10,
            crisis_decay=0.3,
            min_strength=0.05,
            rng=np.random.RandomState(42),
        )

        nm.update_evidence(confidence=0.0, consistency=1.0, dominant_strategy=0)
        assert nm.strength == pytest.approx(0.8)

        # Accumulate exactly 10 anomalies
        for _ in range(10):
            nm.record_anomaly(observed_strategy=1)

        dissolved = nm.check_crisis()
        assert not dissolved  # 0.8 * 0.3 = 0.24 > 0.05
        assert nm.strength == pytest.approx(0.8 * 0.3)
        assert nm.anomaly_count == 0  # reset
        assert nm.has_norm()  # still has norm

    def test_dissolution_when_strength_below_min(self):
        """After crisis, if strength < min_strength, norm dissolves."""
        nm = NormativeMemory(
            ddm_noise=0.0,
            crystal_threshold=1.0,
            initial_strength=0.2,
            crisis_threshold=5,
            crisis_decay=0.3,
            min_strength=0.1,
            rng=np.random.RandomState(42),
        )

        nm.update_evidence(confidence=0.0, consistency=1.0, dominant_strategy=0)
        assert nm.strength == pytest.approx(0.2)

        # 5 anomalies -> crisis -> 0.2 * 0.3 = 0.06 < 0.1 -> dissolve
        for _ in range(5):
            nm.record_anomaly(observed_strategy=1)

        dissolved = nm.check_crisis()
        assert dissolved
        assert not nm.has_norm()
        assert nm.norm is None
        assert nm.evidence == pytest.approx(0.0)  # reset for re-crystallisation

    def test_no_crisis_below_threshold(self):
        """Anomalies below threshold should not trigger crisis."""
        nm = NormativeMemory(
            ddm_noise=0.0,
            crystal_threshold=1.0,
            initial_strength=0.8,
            crisis_threshold=10,
            rng=np.random.RandomState(42),
        )

        nm.update_evidence(confidence=0.0, consistency=1.0, dominant_strategy=0)

        for _ in range(9):  # one less than threshold
            nm.record_anomaly(observed_strategy=1)

        dissolved = nm.check_crisis()
        assert not dissolved
        assert nm.strength == pytest.approx(0.8)  # unchanged
        assert nm.anomaly_count == 9  # unchanged

    def test_crisis_noop_without_norm(self):
        """check_crisis should return False when no norm exists."""
        nm = NormativeMemory(rng=np.random.RandomState(42))
        assert not nm.check_crisis()

    def test_re_crystallisation_after_dissolution(self):
        """After dissolution, agent can form a new norm."""
        nm = NormativeMemory(
            ddm_noise=0.0,
            crystal_threshold=1.0,
            initial_strength=0.2,
            crisis_threshold=1,
            crisis_decay=0.3,
            min_strength=0.1,
            rng=np.random.RandomState(42),
        )

        # Crystallise strategy 0
        nm.update_evidence(confidence=0.0, consistency=1.0, dominant_strategy=0)
        assert nm.norm == 0

        # Dissolve
        nm.record_anomaly(observed_strategy=1)
        dissolved = nm.check_crisis()
        assert dissolved
        assert not nm.has_norm()

        # Re-crystallise with strategy 1
        result = nm.update_evidence(confidence=0.0, consistency=1.0, dominant_strategy=1)
        assert result
        assert nm.norm == 1


class TestCompliance:
    """Tests for compliance = sigma^k (Eq. 13)."""

    def test_compliance_values(self):
        """Verify sigma^k for known sigma values with k=2."""
        nm = NormativeMemory(
            ddm_noise=0.0,
            crystal_threshold=1.0,
            compliance_exponent=2.0,
            rng=np.random.RandomState(42),
        )

        # Need a norm to have non-zero compliance
        # Test via direct state manipulation for precision
        nm._norm = 0

        test_cases = [
            (0.9, 0.81),
            (0.8, 0.64),
            (0.5, 0.25),
            (0.3, 0.09),
            (0.1, 0.01),
        ]

        for sigma, expected_compliance in test_cases:
            nm._strength = sigma
            assert nm.get_compliance() == pytest.approx(expected_compliance, abs=1e-10)

    def test_compliance_zero_without_norm(self):
        """Compliance should be 0 when no norm exists."""
        nm = NormativeMemory(rng=np.random.RandomState(42))
        assert nm.get_compliance() == pytest.approx(0.0)

    def test_compliance_with_different_exponents(self):
        """Test compliance with k=1 and k=3."""
        nm = NormativeMemory(
            ddm_noise=0.0,
            crystal_threshold=1.0,
            compliance_exponent=3.0,
            rng=np.random.RandomState(42),
        )

        nm._norm = 0
        nm._strength = 0.5
        assert nm.get_compliance() == pytest.approx(0.125)  # 0.5^3


class TestNormBelief:
    """Tests for norm belief vector (Eq. 14)."""

    def test_norm_belief_strategy_0(self):
        """Norm=0 -> one_hot = [1, 0]."""
        nm = NormativeMemory(rng=np.random.RandomState(42))
        nm._norm = 0
        belief = nm.get_norm_belief()
        np.testing.assert_array_equal(belief, [1.0, 0.0])

    def test_norm_belief_strategy_1(self):
        """Norm=1 -> one_hot = [0, 1]."""
        nm = NormativeMemory(rng=np.random.RandomState(42))
        nm._norm = 1
        belief = nm.get_norm_belief()
        np.testing.assert_array_equal(belief, [0.0, 1.0])

    def test_norm_belief_none_without_norm(self):
        """No norm -> None."""
        nm = NormativeMemory(rng=np.random.RandomState(42))
        assert nm.get_norm_belief() is None


class TestEnforcement:
    """Tests for violation-triggered enforcement (Eq. 16)."""

    def test_enforcement_requires_all_three_conditions(self):
        """Enforcement needs: has_norm AND sigma > threshold AND violation."""
        nm = NormativeMemory(
            ddm_noise=0.0,
            crystal_threshold=1.0,
            initial_strength=0.8,
            rng=np.random.RandomState(42),
        )

        # No norm -> no enforcement
        assert not nm.should_enforce(observed_strategy=1, enforce_threshold=0.7)

        # Crystallise norm for strategy 0
        nm.update_evidence(confidence=0.0, consistency=1.0, dominant_strategy=0)

        # Has norm (0), high sigma (0.8 > 0.7), conforming -> no enforcement
        assert not nm.should_enforce(observed_strategy=0, enforce_threshold=0.7)

        # Has norm, high sigma, violation -> YES enforcement
        assert nm.should_enforce(observed_strategy=1, enforce_threshold=0.7)

        # Has norm, low sigma (<=0.7), violation -> no enforcement
        nm._strength = 0.7
        assert not nm.should_enforce(observed_strategy=1, enforce_threshold=0.7)

        nm._strength = 0.5
        assert not nm.should_enforce(observed_strategy=1, enforce_threshold=0.7)

    def test_record_enforcement_increments_count(self):
        """Enforcement count tracks signals sent."""
        nm = NormativeMemory(rng=np.random.RandomState(42))
        assert nm.get_state().enforcement_count == 0

        nm.record_enforcement()
        nm.record_enforcement()
        nm.record_enforcement()
        assert nm.get_state().enforcement_count == 3


class TestSignalReception:
    """Tests for normative signal boost (Eq. 17)."""

    def test_signal_boost_applies_then_resets(self):
        """Signal amplification should apply once then reset to 1.0."""
        nm = NormativeMemory(
            ddm_noise=0.0,
            crystal_threshold=10.0,  # high so we don't crystallise
            rng=np.random.RandomState(42),
        )

        # Without signal: drift = (1-0) * 1.0 * 1.0 = 1.0
        nm.update_evidence(confidence=0.0, consistency=1.0, dominant_strategy=0)
        evidence_after_1 = nm.evidence
        assert evidence_after_1 == pytest.approx(1.0)

        # Receive signal with amplification 2.0
        nm.receive_signal(signal_amplification=2.0)

        # Next update: drift = (1-0) * 1.0 * 2.0 = 2.0
        nm.update_evidence(confidence=0.0, consistency=1.0, dominant_strategy=0)
        evidence_after_2 = nm.evidence
        assert evidence_after_2 == pytest.approx(3.0)  # 1.0 + 2.0

        # Third update: boost should have reset, drift = 1.0
        nm.update_evidence(confidence=0.0, consistency=1.0, dominant_strategy=0)
        evidence_after_3 = nm.evidence
        assert evidence_after_3 == pytest.approx(4.0)  # 3.0 + 1.0

    def test_signal_noop_with_existing_norm(self):
        """Signal should not apply if norm already exists."""
        nm = NormativeMemory(
            ddm_noise=0.0,
            crystal_threshold=1.0,
            rng=np.random.RandomState(42),
        )

        # Crystallise
        nm.update_evidence(confidence=0.0, consistency=1.0, dominant_strategy=0)
        assert nm.has_norm()

        # Signal should be ignored
        nm.receive_signal(signal_amplification=5.0)
        assert nm._signal_boost == 1.0  # not changed


class TestStateAndReset:
    """Tests for state snapshots and reset."""

    def test_get_state_returns_correct_snapshot(self):
        """State snapshot should reflect all current values."""
        nm = NormativeMemory(
            ddm_noise=0.0,
            crystal_threshold=1.0,
            initial_strength=0.8,
            compliance_exponent=2.0,
            rng=np.random.RandomState(42),
        )

        nm.update_evidence(confidence=0.0, consistency=1.0, dominant_strategy=0)
        nm.record_anomaly(observed_strategy=1)
        nm.record_enforcement()

        state = nm.get_state()
        assert isinstance(state, NormativeState)
        assert state.norm == 0
        assert state.strength == pytest.approx(0.8)
        assert state.anomaly_count == 1
        assert state.has_norm is True
        assert state.compliance == pytest.approx(0.64)
        assert state.enforcement_count == 1

    def test_reset_clears_all_state(self):
        """Reset should return to initial clean state."""
        nm = NormativeMemory(
            ddm_noise=0.0,
            crystal_threshold=1.0,
            rng=np.random.RandomState(42),
        )

        nm.update_evidence(confidence=0.0, consistency=1.0, dominant_strategy=0)
        nm.record_anomaly(observed_strategy=1)
        nm.record_enforcement()

        nm.reset()

        assert nm.norm is None
        assert nm.strength == pytest.approx(0.0)
        assert nm.anomaly_count == 0
        assert nm.evidence == pytest.approx(0.0)
        assert not nm.has_norm()
        assert nm.get_state().enforcement_count == 0
