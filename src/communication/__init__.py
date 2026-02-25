"""
Communication module (Future Extension).

This module will handle communication between agents, enabling:
- Strategy signaling before interaction
- Reputation sharing
- Norm broadcasting
- Social learning through observation

Planned components:
- BaseCommunication: Abstract interface for communication protocols
- DirectSignaling: Agents can signal intended strategy
- ReputationNetwork: Share information about other agents
- NormBroadcast: Announce and receive norm information

Example future usage:
    from src.communication import create_communication, CommunicationMode

    comm = create_communication(
        mode=CommunicationMode.DIRECT_SIGNALING,
        signal_cost=0.1,
        signal_reliability=0.9
    )

    # Before interaction
    signal = comm.send_signal(my_strategy, partner)
    received = comm.receive_signal(partner)

    # Update beliefs based on communication
    belief = comm.update_belief(belief, received)
"""

from enum import Enum
from abc import ABC, abstractmethod
from typing import Optional, Any


class CommunicationMode(Enum):
    """Available communication modes (placeholder)."""
    NONE = "none"                    # No communication (default)
    DIRECT_SIGNALING = "direct"      # Signal intended strategy
    REPUTATION = "reputation"        # Share reputation information
    NORM_BROADCAST = "broadcast"     # Broadcast norm information


class BaseCommunication(ABC):
    """
    Abstract base class for communication protocols.

    To be implemented when communication features are added.
    """

    @property
    @abstractmethod
    def mode(self) -> CommunicationMode:
        """Return communication mode."""
        pass

    @abstractmethod
    def send_signal(self, strategy: int, partner_id: int) -> Optional[Any]:
        """Send a signal to partner."""
        pass

    @abstractmethod
    def receive_signal(self, partner_id: int) -> Optional[Any]:
        """Receive signal from partner."""
        pass

    @abstractmethod
    def update_belief(self, belief: Any, signal: Optional[Any]) -> Any:
        """Update belief based on received signal."""
        pass


class NoCommunication(BaseCommunication):
    """Default: no communication between agents."""

    @property
    def mode(self) -> CommunicationMode:
        return CommunicationMode.NONE

    def send_signal(self, strategy: int, partner_id: int) -> Optional[Any]:
        return None

    def receive_signal(self, partner_id: int) -> Optional[Any]:
        return None

    def update_belief(self, belief: Any, signal: Optional[Any]) -> Any:
        return belief


def create_communication(
    mode: CommunicationMode = CommunicationMode.NONE,
    **kwargs
) -> BaseCommunication:
    """Factory function for communication protocols."""
    if mode == CommunicationMode.NONE:
        return NoCommunication()
    else:
        raise NotImplementedError(f"Communication mode {mode} not yet implemented")


from .observation import (
    Observation,
    ObservationCollector,
    BaseObservationNetwork,
    FullObservationNetwork,
    LocalObservationNetwork,
    NoObservationNetwork,
)

from .mechanisms import (
    # Normative Signaling (Bicchieri 2006)
    NormativeMessage,
    NormativeSignaling,
    # Pre-play Signaling (Skyrms 2010)
    SignalType,
    PrePlaySignal,
    PrePlaySignaling,
    # Threshold Contagion (Centola 2018)
    ContagionType,
    ThresholdContagion,
    # Combined manager
    CommunicationManager,
)

__all__ = [
    # Legacy interface
    "CommunicationMode",
    "BaseCommunication",
    "NoCommunication",
    "create_communication",
    # Observation-based communication
    "Observation",
    "ObservationCollector",
    "BaseObservationNetwork",
    "FullObservationNetwork",
    "LocalObservationNetwork",
    "NoObservationNetwork",
    # Normative Signaling (Bicchieri 2006)
    "NormativeMessage",
    "NormativeSignaling",
    # Pre-play Signaling (Skyrms 2010)
    "SignalType",
    "PrePlaySignal",
    "PrePlaySignaling",
    # Threshold Contagion (Centola 2018)
    "ContagionType",
    "ThresholdContagion",
    # Combined manager
    "CommunicationManager",
]
