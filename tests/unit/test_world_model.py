"""
TDD Tests for Action, WorldModel, and ML Correction.

Sprint 2 — These tests define the contract for how the engine predicts
state transitions, computes errors, and learns from mistakes.

Tests written BEFORE implementation.
"""

import numpy as np
import pytest

from opl.model.action import Action
from opl.model.rules import simple_stock_rule
from opl.model.world_model import WorldModel
from opl.state.vector import StateVector

# ─── Action Tests ─────────────────────────────────────────────────────────────


class TestAction:
    """Action represents a controllable input the engine can recommend."""

    @pytest.mark.unit
    def test_create_action(self):
        """Action has a name and a numeric value."""
        a = Action(name="reorder", value=30)
        assert a.name == "reorder"
        assert a.value == 30

    @pytest.mark.unit
    def test_action_equality(self):
        """Two actions with same name and value are equal."""
        a = Action(name="reorder", value=30)
        b = Action(name="reorder", value=30)
        assert a == b

    @pytest.mark.unit
    def test_action_inequality(self):
        """Different value means different action."""
        a = Action(name="reorder", value=30)
        b = Action(name="reorder", value=50)
        assert a != b

    @pytest.mark.unit
    def test_action_repr(self):
        """Readable string for logging."""
        a = Action(name="reorder", value=30)
        assert "reorder" in repr(a)
        assert "30" in repr(a)

    @pytest.mark.unit
    def test_no_action(self):
        """A 'do nothing' action is value=0."""
        a = Action.no_op()
        assert a.value == 0


# ─── Rule Tests ───────────────────────────────────────────────────────────────


class TestSimpleStockRule:
    """The base rule encodes domain physics — how stock evolves without ML."""

    @pytest.mark.unit
    def test_stock_decreases_by_demand(self):
        """stock(t+1) = stock(t) - demand(t) + arrivals."""
        # State: [stock=100, demand=20, incoming=50, delay=2]
        s_t = StateVector([100, 20, 50, 2], names=["stock", "demand", "incoming", "delay"])
        action = Action(name="reorder", value=0)  # no reorder
        s_next = simple_stock_rule(s_t, action)
        # stock = 100 - 20 + 0 (delay>0, so no arrival) = 80
        assert s_next[0] == pytest.approx(80.0)

    @pytest.mark.unit
    def test_incoming_arrives_when_delay_zero(self):
        """When delay reaches 0, incoming stock is added."""
        s_t = StateVector([100, 20, 50, 0], names=["stock", "demand", "incoming", "delay"])
        action = Action(name="reorder", value=0)
        s_next = simple_stock_rule(s_t, action)
        # stock = 100 - 20 + 50 = 130
        assert s_next[0] == pytest.approx(130.0)

    @pytest.mark.unit
    def test_delay_decrements(self):
        """Delay ticks down by 1 each step."""
        s_t = StateVector([100, 20, 50, 3], names=["stock", "demand", "incoming", "delay"])
        action = Action(name="reorder", value=0)
        s_next = simple_stock_rule(s_t, action)
        assert s_next[3] == pytest.approx(2.0)  # delay: 3 → 2

    @pytest.mark.unit
    def test_reorder_action_sets_incoming(self):
        """When a reorder action is taken, it sets up incoming stock."""
        s_t = StateVector([100, 20, 0, 0], names=["stock", "demand", "incoming", "delay"])
        action = Action(name="reorder", value=40)
        s_next = simple_stock_rule(s_t, action)
        # After reorder: incoming = 40, delay = default lead time (3)
        assert s_next[2] == pytest.approx(40.0)  # incoming
        assert s_next[3] == pytest.approx(3.0)  # delay (default lead time)

    @pytest.mark.unit
    def test_output_is_state_vector(self):
        """Rule must return a StateVector, not raw array."""
        s_t = StateVector([100, 20, 50, 2], names=["stock", "demand", "incoming", "delay"])
        action = Action(name="reorder", value=0)
        result = simple_stock_rule(s_t, action)
        assert isinstance(result, StateVector)

    @pytest.mark.unit
    def test_demand_stays_constant_in_base_rule(self):
        """Base rule assumes demand persists (naive). ML corrects this."""
        s_t = StateVector([100, 20, 50, 2], names=["stock", "demand", "incoming", "delay"])
        action = Action(name="reorder", value=0)
        s_next = simple_stock_rule(s_t, action)
        assert s_next[1] == pytest.approx(20.0)


# ─── WorldModel Tests ────────────────────────────────────────────────────────


class TestWorldModel:
    """WorldModel = rule-based prediction + optional ML correction."""

    @pytest.mark.unit
    def test_predict_uses_rule(self):
        """Without correction, prediction equals the rule output."""
        model = WorldModel(rule=simple_stock_rule)
        s_t = StateVector([100, 20, 50, 2], names=["stock", "demand", "incoming", "delay"])
        action = Action(name="reorder", value=0)
        s_pred = model.predict(s_t, action)
        s_rule = simple_stock_rule(s_t, action)
        assert s_pred == s_rule

    @pytest.mark.unit
    def test_predict_returns_state_vector(self):
        """Prediction must return a StateVector."""
        model = WorldModel(rule=simple_stock_rule)
        s_t = StateVector([100, 20, 50, 2], names=["stock", "demand", "incoming", "delay"])
        action = Action(name="reorder", value=0)
        result = model.predict(s_t, action)
        assert isinstance(result, StateVector)

    @pytest.mark.unit
    def test_compute_error(self):
        """error = real - predicted, element-wise."""
        predicted = StateVector([80, 20, 50, 1])
        real = StateVector([75, 25, 50, 1])
        error = WorldModel.compute_error(predicted, real)
        assert isinstance(error, StateVector)
        assert list(error.values) == pytest.approx([-5, 5, 0, 0])

    @pytest.mark.unit
    def test_error_log_starts_empty(self):
        """New model has no error history."""
        model = WorldModel(rule=simple_stock_rule)
        assert len(model.error_history) == 0

    @pytest.mark.unit
    def test_record_observation_logs_error(self):
        """After recording an observation, error history grows."""
        model = WorldModel(rule=simple_stock_rule)
        s_t = StateVector([100, 20, 50, 2], names=["stock", "demand", "incoming", "delay"])
        action = Action(name="reorder", value=0)
        s_real = StateVector([78, 22, 50, 1], names=["stock", "demand", "incoming", "delay"])
        model.record_observation(s_t, action, s_real)
        assert len(model.error_history) == 1

    @pytest.mark.unit
    def test_correction_not_trained_initially(self):
        """ML correction is off until explicitly trained."""
        model = WorldModel(rule=simple_stock_rule)
        assert model.correction_trained is False

    @pytest.mark.unit
    def test_train_correction_requires_minimum_observations(self):
        """Cannot train ML correction with too few data points."""
        model = WorldModel(rule=simple_stock_rule)
        # Record only 2 observations (minimum should be ~10)
        for i in range(2):
            s_t = StateVector([100 - i * 10, 20, 50, 2], names=["stock", "demand", "incoming", "delay"])
            action = Action(name="reorder", value=0)
            s_real = StateVector([78 - i * 10, 22, 50, 1], names=["stock", "demand", "incoming", "delay"])
            model.record_observation(s_t, action, s_real)
        with pytest.raises(ValueError, match="observations"):
            model.train_correction()

    @pytest.mark.unit
    def test_train_correction_succeeds_with_enough_data(self):
        """ML correction trains successfully with sufficient observations."""
        model = WorldModel(rule=simple_stock_rule)
        np.random.seed(42)
        for i in range(30):
            stock = 100 - i * 2 + np.random.normal(0, 3)
            demand = 20 + np.random.normal(0, 2)
            s_t = StateVector([stock, demand, 50, 2], names=["stock", "demand", "incoming", "delay"])
            action = Action(name="reorder", value=0)
            # Real outcome has a systematic bias the rule misses
            s_real = StateVector(
                [stock - demand - 3, demand + 1, 50, 1],
                names=["stock", "demand", "incoming", "delay"],
            )
            model.record_observation(s_t, action, s_real)
        model.train_correction()
        assert model.correction_trained is True

    @pytest.mark.unit
    def test_correction_improves_prediction(self):
        """After training, predictions should be closer to reality."""
        model = WorldModel(rule=simple_stock_rule)
        np.random.seed(42)

        # Generate training data with a systematic bias
        for i in range(50):
            stock = 100 - i + np.random.normal(0, 2)
            demand = 20 + np.random.normal(0, 1)
            s_t = StateVector([stock, demand, 0, 0], names=["stock", "demand", "incoming", "delay"])
            action = Action(name="reorder", value=0)
            # Reality: demand is systematically 5 higher than what rule assumes
            real_demand = demand + 5
            s_real = StateVector(
                [stock - real_demand, real_demand, 0, 0],
                names=["stock", "demand", "incoming", "delay"],
            )
            model.record_observation(s_t, action, s_real)

        # Measure error before correction
        test_state = StateVector([80, 20, 0, 0], names=["stock", "demand", "incoming", "delay"])
        test_action = Action(name="reorder", value=0)
        pred_before = model.predict(test_state, test_action)

        # Train and predict with correction
        model.train_correction()
        pred_after = model.predict(test_state, test_action)

        # Real outcome would be stock=80-25=55, demand=25
        real = StateVector([55, 25, 0, 0], names=["stock", "demand", "incoming", "delay"])
        error_before = np.abs((pred_before - real).values).mean()
        error_after = np.abs((pred_after - real).values).mean()

        assert error_after < error_before, (
            f"Correction should reduce error: before={error_before:.2f}, after={error_after:.2f}"
        )
