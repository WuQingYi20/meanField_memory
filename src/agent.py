"""
Agent class integrating memory, decision, and trust systems.

Each agent maintains beliefs about the environment based on memory,
makes decisions using probability matching, and adapts trust based on prediction errors.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any

from .memory import BaseMemory, Interaction, create_memory
from .decision import TrustBasedDecision
from .trust import TrustManager


class Agent:
    """
    An agent in the ABM simulation.

    Simplified architecture with Trust as the central state variable:
    - Memory system: Stores interaction history
    - Trust: Updated based on prediction accuracy
    - Memory window: Determined by Trust (for dynamic memory)
    - Action selection: Probability matching (independent of Trust)

    The feedback loop (cognitive lock-in):
    - Higher prediction accuracy → Higher Trust
    - Higher Trust → Larger memory window
    - Larger window → More stable beliefs
    - More stable beliefs → Harder to reverse norms
    """

    def __init__(
        self,
        agent_id: int,
        memory: BaseMemory,
        decision: TrustBasedDecision,
        trust_manager: Optional[TrustManager] = None,
        initial_strategy: Optional[int] = None,
    ):
        """
        Initialize an agent.

        Args:
            agent_id: Unique identifier
            memory: Memory system instance
            decision: Decision mechanism instance
            trust_manager: Trust manager (optional, creates default if None)
            initial_strategy: Starting strategy (random if None)
        """
        self._id = agent_id
        self._memory = memory
        self._decision = decision
        self._trust_manager = trust_manager or TrustManager()

        # Initialize strategy (random if not specified)
        if initial_strategy is not None:
            self._current_strategy = initial_strategy
        else:
            self._current_strategy = np.random.choice([0, 1])

        # Link dynamic memory to decision mechanism's trust
        self._link_memory_to_trust()

        # Statistics
        self._total_interactions = 0
        self._successful_coordinations = 0
        self._strategy_switches = 0
        self._last_prediction: Optional[int] = None

    def _link_memory_to_trust(self) -> None:
        """Connect dynamic memory's window to decision's trust."""
        from .memory import DynamicMemory

        if isinstance(self._memory, DynamicMemory):
            # Create trust getter from decision mechanism
            trust_getter = lambda: self._decision.trust
            self._memory.set_trust_getter(trust_getter)

    @property
    def id(self) -> int:
        """Agent's unique identifier."""
        return self._id

    @property
    def current_strategy(self) -> int:
        """Agent's current strategy (0 or 1)."""
        return self._current_strategy

    @property
    def trust(self) -> float:
        """Current trust level."""
        return self._decision.trust

    @property
    def belief(self) -> np.ndarray:
        """Current belief about partner strategy distribution."""
        return self._memory.get_strategy_distribution()

    @property
    def memory_window(self) -> int:
        """Current effective memory window size."""
        from .memory import DynamicMemory

        if isinstance(self._memory, DynamicMemory):
            return self._memory.get_effective_window()
        return self._memory.max_size

    def choose_action(self) -> Tuple[int, int]:
        """
        Choose an action based on current beliefs.

        Uses probability matching: P(A) = belief_A
        Also generates a prediction (argmax of belief).

        Returns:
            Tuple of (chosen_action, predicted_partner_action)
        """
        belief = self._memory.get_strategy_distribution()
        action, prediction = self._decision.choose_action(belief)

        # Track for later update
        self._last_prediction = prediction

        # Track strategy switches
        if action != self._current_strategy:
            self._strategy_switches += 1

        self._current_strategy = action
        return action, prediction

    def update(
        self,
        tick: int,
        partner_strategy: int,
        success: bool,
        payoff: float,
    ) -> None:
        """
        Update agent state after an interaction.

        Records interaction in memory and updates trust
        based on prediction accuracy.

        Args:
            tick: Current simulation time
            partner_strategy: Strategy partner chose
            success: Whether coordination was successful
            payoff: Payoff received
        """
        # Create interaction record
        interaction = Interaction(
            tick=tick,
            partner_strategy=partner_strategy,
            own_strategy=self._current_strategy,
            success=success,
            payoff=payoff,
        )

        # Update memory
        self._memory.add(interaction)

        # Update trust based on prediction accuracy
        if self._last_prediction is not None:
            self._decision.update_trust(self._last_prediction, partner_strategy)

        # Update statistics
        self._total_interactions += 1
        if success:
            self._successful_coordinations += 1

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current agent metrics.

        Returns:
            Dict with agent statistics
        """
        return {
            "id": self._id,
            "strategy": self._current_strategy,
            "trust": self.trust,
            "belief": self.belief.tolist(),
            "memory_window": self.memory_window,
            "memory_size": len(self._memory),
            "total_interactions": self._total_interactions,
            "successful_coordinations": self._successful_coordinations,
            "coordination_rate": (
                self._successful_coordinations / self._total_interactions
                if self._total_interactions > 0
                else 0.0
            ),
            "strategy_switches": self._strategy_switches,
            "prediction_accuracy": self._decision.get_prediction_accuracy(),
        }

    def reset(self, initial_strategy: Optional[int] = None) -> None:
        """
        Reset agent to initial state.

        Args:
            initial_strategy: Starting strategy (random if None)
        """
        self._memory.clear()
        self._decision.reset()

        if initial_strategy is not None:
            self._current_strategy = initial_strategy
        else:
            self._current_strategy = np.random.choice([0, 1])

        self._total_interactions = 0
        self._successful_coordinations = 0
        self._strategy_switches = 0
        self._last_prediction = None

    def __repr__(self) -> str:
        return (
            f"Agent(id={self._id}, strategy={self._current_strategy}, "
            f"trust={self.trust:.3f})"
        )


def create_agent(
    agent_id: int,
    memory_type: str = "fixed",
    memory_size: int = 5,
    decay_rate: float = 0.9,
    dynamic_base: int = 2,
    dynamic_max: int = 6,
    initial_trust: float = 0.5,
    alpha: float = 0.1,
    beta: float = 0.3,
    initial_strategy: Optional[int] = None,
) -> Agent:
    """
    Factory function to create an agent with specified configuration.

    Args:
        agent_id: Unique identifier
        memory_type: 'fixed', 'decay', or 'dynamic'
        memory_size: Size for fixed memory
        decay_rate: Lambda for decay memory
        dynamic_base: Base window for dynamic memory
        dynamic_max: Max window for dynamic memory
        initial_trust: Starting trust level
        alpha: Trust increase rate on correct prediction
        beta: Trust decay rate on wrong prediction
        initial_strategy: Starting strategy

    Returns:
        Configured Agent instance
    """
    # Create memory based on type
    if memory_type == "fixed":
        memory = create_memory("fixed", size=memory_size)
    elif memory_type == "decay":
        memory = create_memory("decay", max_size=memory_size * 2, decay_rate=decay_rate)
    elif memory_type == "dynamic":
        memory = create_memory(
            "dynamic",
            max_size=dynamic_max,
            base_size=dynamic_base,
        )
    else:
        raise ValueError(f"Unknown memory type: {memory_type}")

    # Create decision mechanism (simplified: Trust-based)
    decision = TrustBasedDecision(
        initial_trust=initial_trust,
        alpha=alpha,
        beta=beta,
    )

    # Create trust manager
    trust_manager = TrustManager(
        memory_base=dynamic_base,
        memory_max=dynamic_max,
    )

    return Agent(
        agent_id=agent_id,
        memory=memory,
        decision=decision,
        trust_manager=trust_manager,
        initial_strategy=initial_strategy,
    )
