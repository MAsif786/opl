from __future__ import annotations

import numpy as np
from typing import Any, Sequence
from opl.config.schema import DomainConfig
from .vector import StateVector, StateValidationError


class StateBuilder:
    """Utility to safely build StateVectors from raw dictionaries based on config."""

    def __init__(self, config: DomainConfig | None = None, fields: list[str] | None = None):
        if config:
            self.numeric_dims = config.dimensions
            self.categorical_dims = config.categorical_dimensions
        elif fields:
            self.numeric_dims = fields
            self.categorical_dims = []
        else:
            raise ValueError("Must provide either config or fields")

    def build(self, data: dict[str, Any]) -> StateVector:
        """
        Build a StateVector from a raw dictionary.
        - Numerical dimensions are converted to a numpy array.
        - Categorical dimensions are stored as strings in metadata.
        """
        # 1. Process Numerical Data
        vals = []
        for dim in self.numeric_dims:
            val = data.get(dim, 0.0)  # Use 0.0 as a neutral default for schema evolution
            vals.append(float(val))

        # 2. Process Categorical Data
        metadata = {}
        for dim in self.categorical_dims:
            val = data.get(dim)
            if val is not None:
                metadata[dim] = str(val)

        return StateVector(np.array(vals, dtype=np.float64), tuple(self.numeric_dims), metadata)

    def build_series(self, data_list: Sequence[dict[str, Any]]) -> list[StateVector]:
        """Build a list of StateVectors from a sequence of raw dictionaries."""
        return [self.build(row) for row in data_list]
