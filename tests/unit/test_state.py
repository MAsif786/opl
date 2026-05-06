"""
TDD Tests for StateVector and StateBuilder.

Sprint 1 — These tests define the contract for how raw operational data
becomes a numeric state vector that the rest of the engine consumes.

Tests written BEFORE implementation.
"""

import numpy as np
import pytest

from opl.state.builder import StateBuilder
from opl.state.vector import StateValidationError, StateVector

# ─── StateVector Tests ────────────────────────────────────────────────────────


class TestStateVector:
    """StateVector is the atomic data type of the engine — a frozen numeric snapshot."""

    @pytest.mark.unit
    def test_creates_from_list(self):
        """StateVector can be created from a plain list of numbers."""
        sv = StateVector([100, 20, 50, 2])
        assert len(sv) == 4
        assert sv[0] == 100

    @pytest.mark.unit
    def test_creates_from_numpy(self):
        """StateVector can be created from a numpy array."""
        sv = StateVector(np.array([100.0, 20.0, 50.0, 2.0]))
        assert len(sv) == 4
        assert sv[0] == pytest.approx(100.0)

    @pytest.mark.unit
    def test_is_immutable(self):
        """State vectors must be immutable — no accidental mutation of history."""
        sv = StateVector([100, 20, 50, 2])
        with pytest.raises(TypeError):
            sv[0] = 999

    @pytest.mark.unit
    def test_values_returns_numpy_array(self):
        """The .values property exposes a read-only numpy array."""
        sv = StateVector([100, 20, 50, 2])
        assert isinstance(sv.values, np.ndarray)
        assert sv.values.flags.writeable is False

    @pytest.mark.unit
    def test_equality(self):
        """Two StateVectors with same values are equal."""
        a = StateVector([100, 20, 50, 2])
        b = StateVector([100, 20, 50, 2])
        assert a == b

    @pytest.mark.unit
    def test_inequality(self):
        """Two StateVectors with different values are not equal."""
        a = StateVector([100, 20, 50, 2])
        b = StateVector([100, 20, 50, 3])
        assert a != b

    @pytest.mark.unit
    def test_subtraction_returns_state_vector(self):
        """Subtracting two StateVectors returns a new StateVector (for error computation)."""
        a = StateVector([100, 20, 50, 2])
        b = StateVector([80, 25, 50, 2])
        result = a - b
        assert isinstance(result, StateVector)
        assert list(result.values) == [20, -5, 0, 0]

    @pytest.mark.unit
    def test_rejects_empty_input(self):
        """Cannot create a state with no dimensions."""
        with pytest.raises(StateValidationError, match="empty"):
            StateVector([])

    @pytest.mark.unit
    def test_rejects_non_numeric(self):
        """State values must be numeric — no strings or None."""
        with pytest.raises(StateValidationError, match="numeric"):
            StateVector([100, "bad", 50, 2])

    @pytest.mark.unit
    def test_rejects_nan(self):
        """NaN values are invalid — they corrupt simulations."""
        with pytest.raises(StateValidationError, match="NaN"):
            StateVector([100, float("nan"), 50, 2])

    @pytest.mark.unit
    def test_dimension_names(self):
        """StateVector can carry optional dimension labels for explainability."""
        sv = StateVector([100, 20, 50, 2], names=["stock", "demand", "incoming", "delay"])
        assert sv.names == ("stock", "demand", "incoming", "delay")

    @pytest.mark.unit
    def test_to_dict(self):
        """Serialization for logging and debugging."""
        sv = StateVector([100, 20], names=["stock", "demand"])
        d = sv.to_dict()
        assert d == {"stock": 100.0, "demand": 20.0}


# ─── StateBuilder Tests ──────────────────────────────────────────────────────


class TestStateBuilder:
    """StateBuilder converts raw domain data into StateVectors."""

    @pytest.mark.unit
    def test_builds_state_from_raw_dict(self):
        """Given a complete raw data dict, produces a valid StateVector."""
        builder = StateBuilder(
            fields=["stock", "demand", "incoming", "delay"]
        )
        raw = {"stock": 100, "demand": 20, "incoming": 50, "delay": 2}
        state = builder.build(raw)
        assert isinstance(state, StateVector)
        assert list(state.values) == [100.0, 20.0, 50.0, 2.0]

    @pytest.mark.unit
    def test_rejects_missing_required_fields(self):
        """Must fail explicitly if required fields are absent."""
        builder = StateBuilder(
            fields=["stock", "demand", "incoming", "delay"]
        )
        raw = {"stock": 100}  # missing demand, incoming, delay
        with pytest.raises(StateValidationError, match="missing"):
            builder.build(raw)

    @pytest.mark.unit
    def test_field_order_is_deterministic(self):
        """State dimensions must follow the configured field order, not dict order."""
        builder = StateBuilder(fields=["demand", "stock"])
        raw = {"stock": 100, "demand": 20}
        state = builder.build(raw)
        assert state[0] == 20.0   # demand first
        assert state[1] == 100.0  # stock second

    @pytest.mark.unit
    def test_builds_series_from_list_of_dicts(self):
        """Given daily snapshots, produce a list of StateVectors."""
        builder = StateBuilder(fields=["stock", "demand"])
        rows = [
            {"stock": 100, "demand": 20},
            {"stock": 80, "demand": 25},
            {"stock": 55, "demand": 30},
        ]
        series = builder.build_series(rows)
        assert len(series) == 3
        assert all(isinstance(s, StateVector) for s in series)
        assert series[1][0] == 80.0

    @pytest.mark.unit
    def test_extra_fields_are_ignored(self):
        """Raw data may have extra columns — builder picks only configured fields."""
        builder = StateBuilder(fields=["stock"])
        raw = {"stock": 100, "color": "red", "manager": "alice"}
        state = builder.build(raw)
        assert len(state) == 1
        assert state[0] == 100.0

    @pytest.mark.unit
    def test_names_propagate_to_state_vector(self):
        """Field names from builder config become StateVector dimension names."""
        builder = StateBuilder(fields=["stock", "demand"])
        raw = {"stock": 100, "demand": 20}
        state = builder.build(raw)
        assert state.names == ("stock", "demand")
