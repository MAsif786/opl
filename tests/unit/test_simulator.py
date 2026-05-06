"""
TDD Tests for Action Simulator.

Sprint 3 — These tests define the contract for the simulator, which rolls
the state forward N days to evaluate the consequences of different actions.

Tests written BEFORE implementation.
"""

import numpy as np
import pytest

from opl.model.action import Action
from opl.model.rules import simple_stock_rule
from opl.model.world_model import WorldModel
from opl.simulator.rollout import Simulator, Trajectory
from opl.state.vector import StateVector


@pytest.fixture
def base_model():
    """Returns a basic WorldModel with no ML correction."""
    return WorldModel(rule=simple_stock_rule)


@pytest.fixture
def start_state():
    """A standard starting state for tests."""
    return StateVector([100, 20, 0, 0], names=["stock", "demand", "incoming", "delay"])


class TestSimulator:
    """Simulator tests — rolling state forward in time."""

    @pytest.mark.unit
    def test_rollout_length(self, base_model, start_state):
        """Rollout must return exactly N states for horizon N."""
        sim = Simulator(world_model=base_model)
        action = Action.no_op()
        trajectory = sim.rollout(start_state, action, horizon=14)

        assert isinstance(trajectory, Trajectory)
        assert len(trajectory) == 14

    @pytest.mark.unit
    def test_action_only_applied_on_day_one(self, base_model, start_state):
        """The proposed action is applied on day 0, subsequent days use no_op."""
        sim = Simulator(world_model=base_model)
        action = Action(name="reorder", value=50)
        trajectory = sim.rollout(start_state, action, horizon=7)

        # Day 1: action applied (incoming set to 50, delay=3)
        # stock = 100 - 20 = 80
        assert trajectory.states[0][2] == 50.0  # incoming
        assert trajectory.states[0][3] == 3.0  # delay
        assert trajectory.states[0][0] == 80.0  # stock

        # Day 2: no_op applied (incoming stays 50, delay ticks down to 2)
        assert trajectory.states[1][2] == 50.0
        assert trajectory.states[1][3] == 2.0
        assert trajectory.states[1][0] == 60.0  # stock = 80 - 20

    @pytest.mark.unit
    def test_different_actions_produce_different_trajectories(self, base_model, start_state):
        """Taking different actions on day 1 must change the future."""
        sim = Simulator(world_model=base_model)
        t_noop = sim.rollout(start_state, Action.no_op(), horizon=7)
        t_reorder = sim.rollout(start_state, Action("reorder", 100), horizon=7)

        # At day 7 (index 6), stock should be different due to the arrival
        assert t_noop.states[6][0] != t_reorder.states[6][0]
        # Reorder should result in higher stock
        assert t_reorder.states[6][0] > t_noop.states[6][0]

    @pytest.mark.unit
    def test_zero_horizon_raises_error(self, base_model, start_state):
        """Cannot simulate 0 days."""
        sim = Simulator(world_model=base_model)
        with pytest.raises(ValueError, match="Horizon"):
            sim.rollout(start_state, Action.no_op(), horizon=0)

    @pytest.mark.unit
    def test_trajectory_exposes_dimensions_as_series(self, base_model, start_state):
        """Trajectory should easily yield time-series data for a dimension."""
        sim = Simulator(world_model=base_model)
        trajectory = sim.rollout(start_state, Action.no_op(), horizon=5)

        # Stock: 100 -> 80 -> 60 -> 40 -> 20 -> 0
        stock_series = trajectory.get_series("stock")
        assert len(stock_series) == 5
        np.testing.assert_array_almost_equal(stock_series, [80, 60, 40, 20, 0])

    @pytest.mark.unit
    def test_trajectory_detects_stockout(self, base_model, start_state):
        """Helper methods on trajectory can flag bad outcomes."""
        sim = Simulator(world_model=base_model)

        # 100 stock, 20 demand/day, 7 days = stockout on day 6
        t_stockout = sim.rollout(start_state, Action.no_op(), horizon=7)
        assert any(s[0] <= 0 for s in t_stockout.states)

        # Reorder 100 on day 1 (arrives day 4) prevents stockout
        t_safe = sim.rollout(start_state, Action("reorder", 100), horizon=7)
        assert all(s[0] > 0 for s in t_safe.states)
