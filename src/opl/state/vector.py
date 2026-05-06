from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any


class StateValidationError(ValueError):
    """Raised when a StateVector is initialized with invalid data."""
    pass


@dataclass(frozen=True)
class StateVector:
    """
    A hybrid vector representing the state of an entity.
    
    Contains:
    - values: Numerical array for ML and physics calculations.
    - names: Names of the numerical dimensions.
    - metadata: String-based categorical data for context.
    """
    values: np.ndarray
    names: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        # 1. Validation
        if len(self.values) == 0:
            raise StateValidationError("StateVector cannot be empty")
            
        # Convert list/tuple to numpy if needed
        if not isinstance(self.values, np.ndarray):
            try:
                object.__setattr__(self, "values", np.array(self.values, dtype=np.float64))
            except (ValueError, TypeError) as e:
                raise StateValidationError(f"Invalid numeric data: {e}")
        
        if np.isnan(self.values).any():
            raise StateValidationError("StateVector cannot contain NaN values")

        # 2. Immutability
        self.values.flags.writeable = False
        
        # 3. Default names if not provided (stored as tuple)
        if not self.names:
            object.__setattr__(self, "names", tuple(f"dim_{i}" for i in range(len(self.values))))
        else:
            object.__setattr__(self, "names", tuple(self.names))
            
        if len(self.values) != len(self.names):
            raise StateValidationError(f"Values ({len(self.values)}) and names ({len(self.names)}) must have same length")

    @property
    def dimension_names(self) -> tuple[str, ...]:
        """Backward compatibility for tests."""
        return self.names

    def __getitem__(self, key: int | str) -> float:
        """Allow subscripting like a numpy array or by dimension name."""
        if isinstance(key, str):
            if key in self.metadata:
                return self.metadata[key]  # type: ignore
            idx = list(self.names).index(key)
            return self.values[idx]
        return self.values[key]

    def __len__(self) -> int:
        return len(self.values)

    def to_dict(self) -> dict[str, Any]:
        """Merge numerical values and categorical metadata into one dictionary."""
        d = dict(zip(self.names, self.values.tolist()))
        d.update(self.metadata)
        return d

    def __sub__(self, other: StateVector) -> StateVector:
        """Element-wise subtraction of the numerical part."""
        if self.names != other.names:
            raise ValueError("Cannot subtract vectors with different dimensions")
        return StateVector(self.values - other.values, self.names, self.metadata)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, StateVector):
            return False
        return (
            np.array_equal(self.values, other.values) and 
            self.names == other.names and 
            self.metadata == other.metadata
        )

    def __repr__(self) -> str:
        return f"StateVector(num={len(self.values)}, cat={len(self.metadata)})"
