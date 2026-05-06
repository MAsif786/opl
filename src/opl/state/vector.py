"""
StateVector — Immutable numeric snapshot of an entity at a point in time.

This is the atomic data type of the entire engine. Every component
(world model, simulator, evaluator) operates on StateVectors.

Design decisions:
- Immutable: prevents accidental mutation of historical states
- NumPy-backed: fast math for simulation rollouts
- Named dimensions: enables explainability without sacrificing speed
- Strict validation: NaN/empty/non-numeric rejected at creation time
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class StateValidationError(ValueError):
    """Raised when state data fails validation."""


class StateVector:
    """Immutable numeric state vector representing an entity at time t.

    Args:
        values: Numeric values for each state dimension.
        names: Optional dimension labels (e.g., ["stock", "demand"]).

    Raises:
        StateValidationError: If values are empty, non-numeric, or contain NaN.
    """

    __slots__ = ("_values", "_names")

    def __init__(
        self,
        values: Sequence[float] | np.ndarray,
        names: Sequence[str] | None = None,
    ) -> None:
        # Validate non-empty
        if len(values) == 0:
            raise StateValidationError("State vector cannot be empty")

        # Convert to numpy, catching non-numeric
        try:
            arr = np.asarray(values, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise StateValidationError(f"State values must be numeric: {e}") from e

        # Reject NaN
        if np.any(np.isnan(arr)):
            raise StateValidationError("State values must not contain NaN — they corrupt simulations")

        # Freeze the array
        arr.flags.writeable = False
        object.__setattr__(self, "_values", arr)

        # Store names as immutable tuple
        if names is not None:
            if len(names) != len(arr):
                raise StateValidationError(f"Expected {len(arr)} names, got {len(names)}")
            object.__setattr__(self, "_names", tuple(names))
        else:
            object.__setattr__(self, "_names", None)

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def values(self) -> np.ndarray:
        """Read-only numpy array of state values."""
        return self._values

    @property
    def names(self) -> tuple[str, ...] | None:
        """Optional dimension labels."""
        return self._names

    # ── Indexing ──────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, index: int) -> float:
        return float(self._values[index])

    def __setitem__(self, index: int, value: float) -> None:
        raise TypeError("StateVector is immutable")

    # ── Arithmetic ────────────────────────────────────────────────────────

    def __sub__(self, other: StateVector) -> StateVector:
        """Subtraction for error computation: error = real - predicted."""
        if not isinstance(other, StateVector):
            return NotImplemented
        return StateVector(self._values - other._values, names=self._names)

    def __add__(self, other: StateVector) -> StateVector:
        if not isinstance(other, StateVector):
            return NotImplemented
        return StateVector(self._values + other._values, names=self._names)

    # ── Comparison ────────────────────────────────────────────────────────

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StateVector):
            return NotImplemented
        return np.array_equal(self._values, other._values)

    def __hash__(self) -> int:
        return hash(self._values.tobytes())

    # ── Serialization ─────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, float]:
        """Convert to dict. Requires named dimensions."""
        if self._names is None:
            return {str(i): float(v) for i, v in enumerate(self._values)}
        return {name: float(v) for name, v in zip(self._names, self._values)}

    def __repr__(self) -> str:
        if self._names:
            pairs = ", ".join(f"{n}={v:.1f}" for n, v in zip(self._names, self._values))
            return f"StateVector({pairs})"
        return f"StateVector({list(self._values)})"
