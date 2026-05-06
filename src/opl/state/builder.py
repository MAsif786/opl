"""
StateBuilder — Converts raw domain data into StateVectors.

This is the boundary between messy real-world data and the clean
numeric state that the engine operates on. It enforces field presence,
ordering, and type safety.

Design decisions:
- Configured with an ordered list of field names
- Deterministic dimension ordering (not dict-dependent)
- Extra fields silently ignored (real data is always messy)
- Missing required fields are a hard error
"""

from __future__ import annotations

from collections.abc import Sequence

from opl.state.vector import StateValidationError, StateVector


class StateBuilder:
    """Builds StateVectors from raw data dictionaries.

    Args:
        fields: Ordered list of field names to extract from raw data.
                 This defines the state dimensions and their order.
    """

    def __init__(self, fields: list[str]) -> None:
        if not fields:
            raise ValueError("StateBuilder requires at least one field")
        self._fields = list(fields)

    @property
    def fields(self) -> list[str]:
        """Configured field names."""
        return list(self._fields)

    def build(self, raw: dict) -> StateVector:
        """Build a single StateVector from a raw data dictionary.

        Args:
            raw: Dictionary with at least the configured field keys.

        Returns:
            StateVector with values in the configured field order.

        Raises:
            StateValidationError: If required fields are missing.
        """
        missing = [f for f in self._fields if f not in raw]
        if missing:
            raise StateValidationError(
                f"Raw data missing required fields: {missing}"
            )

        values = [raw[f] for f in self._fields]
        return StateVector(values, names=self._fields)

    def build_series(self, rows: Sequence[dict]) -> list[StateVector]:
        """Build a list of StateVectors from a sequence of raw data dicts.

        Args:
            rows: List of daily snapshots, each a dict with field keys.

        Returns:
            List of StateVectors, one per row.
        """
        return [self.build(row) for row in rows]
