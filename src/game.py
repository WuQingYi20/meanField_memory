"""
Coordination game implementation.

Defines the pure coordination game where agents receive payoff
only when they choose the same strategy.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class GameOutcome:
    """
    Result of a game interaction.

    Attributes:
        strategy1: Strategy chosen by player 1
        strategy2: Strategy chosen by player 2
        payoff1: Payoff received by player 1
        payoff2: Payoff received by player 2
        success: Whether coordination was successful
    """

    strategy1: int
    strategy2: int
    payoff1: float
    payoff2: float
    success: bool


class CoordinationGame:
    """
    Pure Coordination Game with 2 strategies.

    Payoff matrix:
        |   | A | B |
        | A | 1 | 0 |
        | B | 0 | 1 |

    Both players receive payoff only when they coordinate
    on the same strategy. The game is symmetric.

    This models situations where agreement matters more
    than which option is chosen (e.g., driving side, meeting places).
    """

    def __init__(
        self,
        coordination_payoff: float = 1.0,
        miscoordination_payoff: float = 0.0,
        num_strategies: int = 2,
    ):
        """
        Initialize coordination game.

        Args:
            coordination_payoff: Payoff when both choose same strategy
            miscoordination_payoff: Payoff when strategies differ
            num_strategies: Number of strategies (fixed at 2)
        """
        if num_strategies != 2:
            raise ValueError("Currently only 2-strategy games are supported")

        self._num_strategies = num_strategies
        self._coord_payoff = coordination_payoff
        self._miscoord_payoff = miscoordination_payoff

        # Build payoff matrix
        self._payoff_matrix = self._build_payoff_matrix()

    def _build_payoff_matrix(self) -> np.ndarray:
        """
        Construct the payoff matrix.

        Returns:
            2x2 matrix where [i,j] is payoff for choosing i when partner chooses j
        """
        matrix = np.full(
            (self._num_strategies, self._num_strategies),
            self._miscoord_payoff
        )
        # Diagonal elements (coordination) get coordination payoff
        np.fill_diagonal(matrix, self._coord_payoff)
        return matrix

    @property
    def num_strategies(self) -> int:
        """Number of available strategies."""
        return self._num_strategies

    @property
    def payoff_matrix(self) -> np.ndarray:
        """The game payoff matrix."""
        return self._payoff_matrix.copy()

    @property
    def coordination_payoff(self) -> float:
        """Payoff for successful coordination."""
        return self._coord_payoff

    @property
    def miscoordination_payoff(self) -> float:
        """Payoff for failed coordination."""
        return self._miscoord_payoff

    def get_payoff(self, my_strategy: int, partner_strategy: int) -> float:
        """
        Get payoff for a strategy pair.

        Args:
            my_strategy: My chosen strategy (0 or 1)
            partner_strategy: Partner's chosen strategy (0 or 1)

        Returns:
            Payoff value
        """
        return self._payoff_matrix[my_strategy, partner_strategy]

    def is_coordination_success(self, strategy1: int, strategy2: int) -> bool:
        """
        Check if two strategies constitute successful coordination.

        Args:
            strategy1: First player's strategy
            strategy2: Second player's strategy

        Returns:
            True if strategies match
        """
        return strategy1 == strategy2

    def play(self, strategy1: int, strategy2: int) -> GameOutcome:
        """
        Execute a game between two strategies.

        Args:
            strategy1: Strategy chosen by player 1
            strategy2: Strategy chosen by player 2

        Returns:
            GameOutcome with payoffs and success indicator
        """
        if not (0 <= strategy1 < self._num_strategies):
            raise ValueError(f"Invalid strategy1: {strategy1}")
        if not (0 <= strategy2 < self._num_strategies):
            raise ValueError(f"Invalid strategy2: {strategy2}")

        success = self.is_coordination_success(strategy1, strategy2)
        payoff1 = self.get_payoff(strategy1, strategy2)
        payoff2 = self.get_payoff(strategy2, strategy1)  # Symmetric

        return GameOutcome(
            strategy1=strategy1,
            strategy2=strategy2,
            payoff1=payoff1,
            payoff2=payoff2,
            success=success,
        )

    def get_strategy_name(self, strategy: int) -> str:
        """
        Get human-readable name for a strategy.

        Args:
            strategy: Strategy index

        Returns:
            Strategy name ('A' or 'B')
        """
        names = ['A', 'B']
        return names[strategy] if strategy < len(names) else f"S{strategy}"

    def describe(self) -> str:
        """Get a text description of the game."""
        return (
            f"Pure Coordination Game\n"
            f"Strategies: {self._num_strategies}\n"
            f"Coordination payoff: {self._coord_payoff}\n"
            f"Miscoordination payoff: {self._miscoord_payoff}\n"
            f"Payoff matrix:\n{self._payoff_matrix}"
        )

    def __repr__(self) -> str:
        return (
            f"CoordinationGame(coord={self._coord_payoff}, "
            f"miscoord={self._miscoord_payoff})"
        )
