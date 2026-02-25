"""
Norm detection module with multi-level definitions.

V5 6-level hierarchy:
- Level 0: NONE - No regularity
- Level 1: BEHAVIORAL - Behavioural regularity (>= 95% adoption, stable 50 ticks)
- Level 2: EMPIRICAL - + Accurate empirical expectations (belief error < 0.10)
- Level 3: SHARED - + Belief consensus (belief variance < 0.05)
- Level 4: NORMATIVE - + Normative rule internalised (>= 80% agents have norm)
- Level 5: INSTITUTIONAL - + Self-enforcing stability (maintained 200+ ticks)

Based on Bicchieri (2006) and Lewis (1969).
"""

import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import IntEnum


class NormLevel(IntEnum):
    """Levels of norm emergence (V5: 6 levels)."""
    NONE = 0           # No norm
    BEHAVIORAL = 1     # Behavioural regularity only
    EMPIRICAL = 2      # + Accurate empirical beliefs (renamed from COGNITIVE)
    SHARED = 3         # + Similar beliefs across agents
    NORMATIVE = 4      # + Normative rule internalised (V5 new)
    INSTITUTIONAL = 5  # + Self-enforcing stability (V5: 200+ ticks)


# Backward compatibility alias
COGNITIVE = NormLevel.EMPIRICAL


@dataclass
class NormState:
    """Complete state of norm detection."""
    # Basic
    level: NormLevel
    tick: int

    # Level 1: Behavioral
    majority_strategy: Optional[int]
    majority_fraction: float
    behavioral_stability: int  # ticks at current majority

    # Level 2: Empirical (belief accuracy)
    mean_belief_error: float  # |agent_belief - true_distribution|

    # Level 3: Shared (belief alignment)
    belief_variance: float  # variance across agents

    # Level 4: Meta (common knowledge proxy)
    meta_accuracy: float  # do agents know what others believe?

    # V5: Normative adoption
    norm_adoption_rate: float = 0.0  # fraction of agents with crystallised norms

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level.value,
            "level_name": self.level.name,
            "tick": self.tick,
            "majority_strategy": self.majority_strategy,
            "majority_fraction": self.majority_fraction,
            "behavioral_stability": self.behavioral_stability,
            "mean_belief_error": self.mean_belief_error,
            "belief_variance": self.belief_variance,
            "meta_accuracy": self.meta_accuracy,
            "norm_adoption_rate": self.norm_adoption_rate,
        }


class NormDetector:
    """
    Multi-level norm detector.

    Tracks the emergence of norms at different levels of
    social/cognitive integration.

    V5: 6 levels with normative adoption and institutional stability.

    Theoretical basis:
    - Bicchieri (2006): empirical & normative expectations
    - Lewis (1969): common knowledge requirement
    - Aoki (2001): shared cognitive constructs
    """

    def __init__(
        self,
        # Level 1 thresholds
        behavioral_threshold: float = 0.95,
        stability_window: int = 50,
        # Level 2 thresholds
        belief_error_threshold: float = 0.1,
        # Level 3 thresholds
        belief_variance_threshold: float = 0.05,
        # Level 4 thresholds
        meta_accuracy_threshold: float = 0.8,
        norm_adoption_threshold: float = 0.8,
        # Level 5 thresholds
        institutional_stability_window: int = 200,
    ):
        """
        Initialize norm detector.

        Args:
            behavioral_threshold: Fraction for behavioral norm (Level 1)
            stability_window: Ticks to maintain threshold for stability
            belief_error_threshold: Max mean error for empirical norm (Level 2)
            belief_variance_threshold: Max variance for shared norm (Level 3)
            meta_accuracy_threshold: Min accuracy for meta-knowledge
            norm_adoption_threshold: Min fraction of agents with norms (Level 4)
            institutional_stability_window: Ticks to sustain Level 4 for Level 5
        """
        self._behavioral_threshold = behavioral_threshold
        self._stability_window = stability_window
        self._belief_error_threshold = belief_error_threshold
        self._belief_variance_threshold = belief_variance_threshold
        self._meta_accuracy_threshold = meta_accuracy_threshold
        self._norm_adoption_threshold = norm_adoption_threshold
        self._institutional_stability_window = institutional_stability_window

        # Tracking state
        self._stability_counter = 0
        self._last_majority = None
        self._normative_stability_counter = 0  # V5: tracks Level 4 duration
        self._emergence_tick: Dict[NormLevel, Optional[int]] = {
            level: None for level in NormLevel if level != NormLevel.NONE
        }

    def detect(
        self,
        strategies: List[int],
        beliefs: List[np.ndarray],
        tick: int,
        meta_beliefs: Optional[List[np.ndarray]] = None,
        norm_adoption_rate: float = 0.0,
    ) -> NormState:
        """
        Detect norm state at current tick.

        Args:
            strategies: Current strategy of each agent (0 or 1)
            beliefs: Each agent's belief about population distribution
            tick: Current simulation tick
            meta_beliefs: Each agent's belief about others' beliefs (optional)
            norm_adoption_rate: Fraction of agents with crystallised norms (V5)

        Returns:
            NormState with detection results
        """
        n = len(strategies)

        # === Level 1: Behavioral regularity ===
        count_0 = strategies.count(0)
        count_1 = n - count_0

        if count_0 >= count_1:
            majority_strategy = 0
            majority_fraction = count_0 / n
        else:
            majority_strategy = 1
            majority_fraction = count_1 / n

        # Track stability
        if majority_strategy == self._last_majority and majority_fraction >= self._behavioral_threshold:
            self._stability_counter += 1
        else:
            self._stability_counter = 0
            self._last_majority = majority_strategy

        behavioral_norm = (
            majority_fraction >= self._behavioral_threshold and
            self._stability_counter >= self._stability_window
        )

        # === Level 2: Empirical belief accuracy ===
        true_distribution = np.array([count_0 / n, count_1 / n])
        belief_errors = [
            np.abs(belief - true_distribution).mean()
            for belief in beliefs
        ]
        mean_belief_error = np.mean(belief_errors)

        empirical_norm = (
            behavioral_norm and
            mean_belief_error < self._belief_error_threshold
        )

        # === Level 3: Shared beliefs ===
        # Stack beliefs and compute variance
        belief_matrix = np.array([b[majority_strategy] for b in beliefs])
        belief_variance = np.var(belief_matrix)

        shared_norm = (
            empirical_norm and
            belief_variance < self._belief_variance_threshold
        )

        # === Level 4: Normative (V5) ===
        # Meta-belief accuracy (used for Level 4+ but computed for backward compat)
        if meta_beliefs is not None and len(meta_beliefs) > 0:
            actual_mean_belief = np.mean(belief_matrix)
            meta_errors = [
                abs(mb[majority_strategy] - actual_mean_belief)
                for mb in meta_beliefs
            ]
            meta_accuracy = 1 - np.mean(meta_errors)
        else:
            meta_accuracy = 1 - min(1.0, belief_variance * 10)

        normative_norm = (
            shared_norm and
            norm_adoption_rate >= self._norm_adoption_threshold
        )

        # === Level 5: Institutional (V5) ===
        if normative_norm:
            self._normative_stability_counter += 1
        else:
            self._normative_stability_counter = 0

        institutional_norm = (
            normative_norm and
            self._normative_stability_counter >= self._institutional_stability_window
        )

        # Determine level
        if institutional_norm:
            level = NormLevel.INSTITUTIONAL
        elif normative_norm:
            level = NormLevel.NORMATIVE
        elif shared_norm:
            level = NormLevel.SHARED
        elif empirical_norm:
            level = NormLevel.EMPIRICAL
        elif behavioral_norm:
            level = NormLevel.BEHAVIORAL
        else:
            level = NormLevel.NONE

        # Track emergence ticks
        for check_level in [NormLevel.BEHAVIORAL, NormLevel.EMPIRICAL,
                           NormLevel.SHARED, NormLevel.NORMATIVE,
                           NormLevel.INSTITUTIONAL]:
            if level.value >= check_level.value and self._emergence_tick[check_level] is None:
                self._emergence_tick[check_level] = tick

        return NormState(
            level=level,
            tick=tick,
            majority_strategy=majority_strategy if level != NormLevel.NONE else None,
            majority_fraction=majority_fraction,
            behavioral_stability=self._stability_counter,
            mean_belief_error=mean_belief_error,
            belief_variance=belief_variance,
            meta_accuracy=meta_accuracy,
            norm_adoption_rate=norm_adoption_rate,
        )

    def get_emergence_ticks(self) -> Dict[str, Optional[int]]:
        """Get tick when each norm level was first reached."""
        return {
            level.name: tick
            for level, tick in self._emergence_tick.items()
        }

    def reset(self) -> None:
        """Reset detector state."""
        self._stability_counter = 0
        self._last_majority = None
        self._normative_stability_counter = 0
        self._emergence_tick = {
            level: None for level in NormLevel if level != NormLevel.NONE
        }

    def __repr__(self) -> str:
        return (
            f"NormDetector(behavioral={self._behavioral_threshold:.0%}, "
            f"stability={self._stability_window})"
        )
