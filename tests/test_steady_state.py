"""
Steady-state verification tests.

Verifies analytical predictions match simulation outcomes:
- Trust converges to C* = pα / (pα + (1-p)β) for known p values
- Window mapping: w = base + floor(C * (max - base))
"""

import numpy as np
import pytest

from src.decision.cognitive_lockin import CognitiveLockInDecision
from src.memory.dynamic import DynamicMemory


class TestTrustSteadyState:
    """Trust should converge to C* = pα/(pα + (1-p)β).

    Note: The update rules (additive increase, multiplicative decay) create
    a stochastic IFS with non-trivial variance. We use time-averaged trust
    over the stationary portion of a long run for reliable comparison.
    """

    @pytest.mark.parametrize("p,expected_star", [
        (0.5, 0.25),    # pα/(pα+(1-p)β) = 0.05/(0.05+0.15) = 0.25
        (0.8, 0.5714),  # 0.08/(0.08+0.06) ≈ 0.5714
        (0.9, 0.75),    # 0.09/(0.09+0.03) = 0.75
    ])
    def test_trust_converges_to_analytical_steady_state(self, p, expected_star):
        """
        Simulate the trust update rule and verify the time-averaged
        trust converges to C* = pα / (pα + (1-p)β).
        """
        alpha, beta = 0.1, 0.3
        decision = CognitiveLockInDecision(
            initial_trust=0.5,
            alpha=alpha,
            beta=beta,
        )

        rng = np.random.RandomState(42)

        # Burn-in: reach stationarity
        for _ in range(5000):
            correct = rng.random() < p
            if correct:
                decision.update(predicted=0, observed=0)
            else:
                decision.update(predicted=0, observed=1)

        # Collect time average over stationary distribution
        trust_samples = []
        for _ in range(10000):
            correct = rng.random() < p
            if correct:
                decision.update(predicted=0, observed=0)
            else:
                decision.update(predicted=0, observed=1)
            trust_samples.append(decision.trust)

        avg_trust = np.mean(trust_samples)

        # Formula verification
        formula_star = decision.get_trust_steady_state(p)
        assert abs(formula_star - expected_star) < 0.01, (
            f"Formula mismatch: got {formula_star:.4f}, expected {expected_star:.4f}"
        )

        # Time-averaged trust should match analytical prediction (ε=0.05)
        assert abs(avg_trust - expected_star) < 0.05, (
            f"Time-averaged trust did not converge: got {avg_trust:.4f}, "
            f"expected {expected_star:.4f} (p={p})"
        )

    def test_trust_steady_state_symmetric(self):
        """With α=β, steady state should be T* = p."""
        alpha = beta = 0.2
        decision = CognitiveLockInDecision(
            initial_trust=0.5,
            alpha=alpha,
            beta=beta,
        )

        p = 0.7
        rng = np.random.RandomState(42)

        # Burn-in
        for _ in range(5000):
            correct = rng.random() < p
            if correct:
                decision.update(predicted=0, observed=0)
            else:
                decision.update(predicted=0, observed=1)

        # Collect time average
        trust_samples = []
        for _ in range(10000):
            correct = rng.random() < p
            if correct:
                decision.update(predicted=0, observed=0)
            else:
                decision.update(predicted=0, observed=1)
            trust_samples.append(decision.trust)

        avg_trust = np.mean(trust_samples)

        # With symmetric α=β: T* = pα/(pα+(1-p)α) = p
        expected = p
        assert abs(avg_trust - expected) < 0.05

    def test_trust_starting_from_different_initials(self):
        """Trust should converge to same C* regardless of starting point."""
        alpha, beta = 0.1, 0.3
        p = 0.8
        expected_star = (p * alpha) / (p * alpha + (1 - p) * beta)

        averages = []
        for initial in [0.01, 0.5, 0.99]:
            decision = CognitiveLockInDecision(
                initial_trust=initial,
                alpha=alpha,
                beta=beta,
            )
            rng = np.random.RandomState(42)

            # Burn-in
            for _ in range(5000):
                correct = rng.random() < p
                if correct:
                    decision.update(predicted=0, observed=0)
                else:
                    decision.update(predicted=0, observed=1)

            # Collect time average
            trust_samples = []
            for _ in range(10000):
                correct = rng.random() < p
                if correct:
                    decision.update(predicted=0, observed=0)
                else:
                    decision.update(predicted=0, observed=1)
                trust_samples.append(decision.trust)

            averages.append(np.mean(trust_samples))

        # All starting points should converge to similar value
        for avg in averages:
            assert abs(avg - expected_star) < 0.05, (
                f"Failed to converge: avg={avg:.4f}, expected={expected_star:.4f}"
            )

        # All averages should be within 0.05 of each other
        spread = max(averages) - min(averages)
        assert spread < 0.05, f"Spread too large: {spread:.4f}"


class TestWindowMapping:
    """Verify w = base + floor(C * (max - base))."""

    @pytest.mark.parametrize("trust,expected_window", [
        (0.0, 2),   # base only
        (0.25, 3),  # 2 + floor(0.25*4) = 2+1 = 3
        (0.5, 4),   # 2 + floor(0.5*4) = 2+2 = 4
        (0.75, 5),  # 2 + floor(0.75*4) = 2+3 = 5
        (1.0, 6),   # 2 + floor(1.0*4) = 2+4 = 6
    ])
    def test_window_mapping_known_values(self, trust, expected_window):
        """Window should follow w = base + floor(C * (max - base))."""
        memory = DynamicMemory(max_size=6, base_size=2)
        memory.set_internal_trust(trust)

        window = memory.get_effective_window()
        assert window == expected_window, (
            f"Trust={trust}: expected window {expected_window}, got {window}"
        )

    def test_window_at_boundaries(self):
        """Window should be base at trust=0 and max at trust=1."""
        memory = DynamicMemory(max_size=6, base_size=2)

        memory.set_internal_trust(0.0)
        assert memory.get_effective_window() == 2

        memory.set_internal_trust(1.0)
        assert memory.get_effective_window() == 6

    def test_window_monotonically_increasing(self):
        """Window should never decrease as trust increases."""
        memory = DynamicMemory(max_size=6, base_size=2)

        prev_window = 0
        for trust_100 in range(101):
            trust = trust_100 / 100.0
            memory.set_internal_trust(trust)
            window = memory.get_effective_window()
            assert window >= prev_window, (
                f"Window decreased at trust={trust}: {window} < {prev_window}"
            )
            prev_window = window
