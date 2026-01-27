"""
Memory systems for ABM simulation.

This module provides different memory mechanisms:
- FixedMemory: Fixed-length sliding window
- DecayMemory: Exponentially decaying weights
- DynamicMemory: Trust-linked adaptive window
"""

from .base import BaseMemory, Interaction
from .fixed import FixedMemory
from .decay import DecayMemory
from .dynamic import DynamicMemory

__all__ = [
    "BaseMemory",
    "Interaction",
    "FixedMemory",
    "DecayMemory",
    "DynamicMemory",
]


def create_memory(memory_type: str, **kwargs):
    """
    Factory function to create memory instances.

    Args:
        memory_type: One of 'fixed', 'decay', 'dynamic'
        **kwargs: Memory-specific parameters

    Returns:
        BaseMemory instance
    """
    memory_classes = {
        "fixed": FixedMemory,
        "decay": DecayMemory,
        "dynamic": DynamicMemory,
    }

    if memory_type not in memory_classes:
        raise ValueError(
            f"Unknown memory type: {memory_type}. "
            f"Available: {list(memory_classes.keys())}"
        )

    return memory_classes[memory_type](**kwargs)
