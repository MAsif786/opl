"""
Action — A controllable input the engine can recommend.

Actions are the levers the decision engine can pull. In logistics,
this is "reorder 30 units". In finance, "allocate $10k". In manufacturing,
"schedule job X on machine Y".

Design decisions:
- Immutable dataclass: actions are proposals, not mutable state
- Named + valued: the name identifies the type, the value its magnitude
- no_op() factory: explicit "do nothing" is a valid decision
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Action:
    """An action the engine can recommend.

    Args:
        name: Type of action (e.g., "reorder", "transfer", "allocate").
        value: Numeric magnitude (e.g., quantity to reorder).
    """

    name: str
    value: float

    @classmethod
    def no_op(cls) -> Action:
        """Create a 'do nothing' action."""
        return cls(name="no_op", value=0)

    def __repr__(self) -> str:
        return f"Action({self.name}={self.value})"
