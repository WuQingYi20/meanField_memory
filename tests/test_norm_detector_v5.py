"""
Tests for V5 NormDetector: 6-level hierarchy, NORMATIVE level,
INSTITUTIONAL level, and backward compatibility.
"""

import numpy as np
import pytest

from src.norms.detector import NormDetector, NormLevel, NormState, COGNITIVE


class TestNormLevelDefinitions:
    """Tests for norm level enum and backward compat."""

    def test_six_levels_exist(self):
        """NormLevel should have 6 levels (0-5)."""
        assert NormLevel.NONE == 0
        assert NormLevel.BEHAVIORAL == 1
        assert NormLevel.EMPIRICAL == 2
        assert NormLevel.SHARED == 3
        assert NormLevel.NORMATIVE == 4
        assert NormLevel.INSTITUTIONAL == 5

    def test_backward_compat_cognitive_alias(self):
        """COGNITIVE should alias to EMPIRICAL (=2)."""
        assert COGNITIVE == NormLevel.EMPIRICAL
        assert COGNITIVE == 2

    def test_intnum_comparison(self):
        """NormLevel should support integer comparison (IntEnum)."""
        assert NormLevel.BEHAVIORAL < NormLevel.EMPIRICAL
        assert NormLevel.NORMATIVE > NormLevel.SHARED
        assert NormLevel.INSTITUTIONAL == 5


class TestLevel4Normative:
    """Tests for Level 4: NORMATIVE (requires >= 80% norm adoption)."""

    def test_level4_requires_norm_adoption(self):
        """Level 4 requires >= 80% norm adoption rate."""
        detector = NormDetector(
            behavioral_threshold=0.95,
            stability_window=1,  # quick for testing
            belief_error_threshold=0.1,
            belief_variance_threshold=0.05,
            norm_adoption_threshold=0.8,
        )

        # 100% strategy 0, accurate beliefs, low variance
        n = 100
        strategies = [0] * n
        beliefs = [np.array([1.0, 0.0])] * n

        # First call to establish stability counter
        detector.detect(strategies, beliefs, 0, norm_adoption_rate=0.9)

        # Second call with norm_adoption_rate = 90% -> Level 4
        state = detector.detect(strategies, beliefs, 1, norm_adoption_rate=0.9)
        assert state.level == NormLevel.NORMATIVE

    def test_level4_not_reached_below_threshold(self):
        """Level 4 should NOT be reached with < 80% adoption."""
        detector = NormDetector(
            behavioral_threshold=0.95,
            stability_window=1,
            belief_error_threshold=0.1,
            belief_variance_threshold=0.05,
            norm_adoption_threshold=0.8,
        )

        n = 100
        strategies = [0] * n
        beliefs = [np.array([1.0, 0.0])] * n

        # Build stability
        detector.detect(strategies, beliefs, 0, norm_adoption_rate=0.5)
        state = detector.detect(strategies, beliefs, 1, norm_adoption_rate=0.5)

        # Should reach Level 3 (SHARED) but not Level 4
        assert state.level <= NormLevel.SHARED


class TestLevel5Institutional:
    """Tests for Level 5: INSTITUTIONAL (requires 200 ticks sustained at Level 4)."""

    def test_level5_requires_sustained_level4(self):
        """Level 5 requires 200 ticks sustained at Level 4."""
        detector = NormDetector(
            behavioral_threshold=0.95,
            stability_window=1,
            belief_error_threshold=0.1,
            belief_variance_threshold=0.05,
            norm_adoption_threshold=0.8,
            institutional_stability_window=200,
        )

        n = 100
        strategies = [0] * n
        beliefs = [np.array([1.0, 0.0])] * n

        # Run 201 ticks at Level 4 conditions
        for t in range(202):
            state = detector.detect(strategies, beliefs, t, norm_adoption_rate=0.9)

        assert state.level == NormLevel.INSTITUTIONAL

    def test_level5_not_reached_before_200_ticks(self):
        """Level 5 should NOT be reached before 200 ticks of Level 4."""
        detector = NormDetector(
            behavioral_threshold=0.95,
            stability_window=1,
            belief_error_threshold=0.1,
            belief_variance_threshold=0.05,
            norm_adoption_threshold=0.8,
            institutional_stability_window=200,
        )

        n = 100
        strategies = [0] * n
        beliefs = [np.array([1.0, 0.0])] * n

        # Run only 100 ticks
        for t in range(100):
            state = detector.detect(strategies, beliefs, t, norm_adoption_rate=0.9)

        # Should be Level 4, not 5
        assert state.level == NormLevel.NORMATIVE

    def test_level5_resets_on_drop(self):
        """Dropping below Level 4 should reset the institutional counter."""
        detector = NormDetector(
            behavioral_threshold=0.95,
            stability_window=1,
            belief_error_threshold=0.1,
            belief_variance_threshold=0.05,
            norm_adoption_threshold=0.8,
            institutional_stability_window=200,
        )

        n = 100
        strategies = [0] * n
        beliefs = [np.array([1.0, 0.0])] * n

        # 150 ticks at Level 4
        for t in range(150):
            detector.detect(strategies, beliefs, t, norm_adoption_rate=0.9)

        # Drop: norm adoption rate too low
        detector.detect(strategies, beliefs, 150, norm_adoption_rate=0.5)

        # Another 100 ticks at Level 4 -> total only 100, not 200
        for t in range(151, 252):
            state = detector.detect(strategies, beliefs, t, norm_adoption_rate=0.9)

        # Should be Level 4, not 5 (counter was reset at tick 150)
        assert state.level == NormLevel.NORMATIVE


class TestBackwardCompatDetection:
    """Tests for backward compatibility of detect() without normative params."""

    def test_detect_without_norm_adoption_rate(self):
        """detect() should work without norm_adoption_rate (defaults to 0.0)."""
        detector = NormDetector()

        strategies = [0] * 100
        beliefs = [np.array([1.0, 0.0])] * 100

        # Should not raise
        state = detector.detect(strategies, beliefs, 0)
        assert state.norm_adoption_rate == 0.0
