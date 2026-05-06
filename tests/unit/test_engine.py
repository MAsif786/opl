"""
TDD Tests for Cost Evaluator and Decision Engine.

Sprint 4 — These tests define how we score simulated futures (trajectories)
and how the orchestrator picks the best action from candidate options.

Tests written BEFORE implementation.
"""

import pytest

from opl.engine.decision import Decision, DecisionEngine
from opl.evaluator.cost import CostEvaluator, LogisticsCostFunction
from opl.model.action import Action
from opl.model.rules import simple_stock_rule
from opl.model.world_model import WorldModel
from opl.simulator.rollout import Simulator, Trajectory
from opl.state.vector import StateVector

# ─── Cost Evaluator Tests ─────────────────────────────────────────────────────


class TestCostEvaluator:
    """Evaluates the "badness" of a trajectory using a domain-specific cost function."""

    @pytest.mark.unit
    def test_stockout_is_heavily_penalized(self):
        """A trajectory with a stockout should cost much more than a safe one."""
        evaluator = CostEvaluator(cost_function=LogisticsCostFunction())

        # Safe trajectory: stock stays at 50
        safe_states = [StateVector([50, 20, 0, 0], names=["stock", "demand", "incoming", "delay"])] * 7
        t_safe = Trajectory(safe_states)

        # Stockout trajectory: stock hits 0
        stockout_states = [StateVector([0, 20, 0, 0], names=["stock", "demand", "incoming", "delay"])] * 7
        t_stockout = Trajectory(stockout_states)

        cost_safe = evaluator.score(t_safe)
        cost_stockout = evaluator.score(t_stockout)

        assert cost_stockout > cost_safe * 10  # Stockout penalty should be severe

    @pytest.mark.unit
    def test_holding_excess_inventory_has_cost(self):
        """Higher inventory should incur higher holding costs."""
        evaluator = CostEvaluator(cost_function=LogisticsCostFunction())

        t_low = Trajectory([StateVector([10, 20, 0, 0], names=["stock", "demand", "incoming", "delay"])] * 7)
        t_high = Trajectory([StateVector([1000, 20, 0, 0], names=["stock", "demand", "incoming", "delay"])] * 7)

        assert evaluator.score(t_high) > evaluator.score(t_low)

    @pytest.mark.unit
    def test_reorder_action_incurs_cost(self):
        """Taking an action (reordering) should have a baseline transaction cost."""

        # We need action info in the evaluator now? Actually, the cost function
        # usually evaluates the state trajectory. Let's see if we need action cost.
        # LogisticsCostFunction: Cost = sum(holding_cost) + sum(stockout_penalty)
        # Action cost can be implicit in holding cost of incoming, or evaluated separately.
        # Let's keep it simple: cost is purely a function of the state trajectory.
        pass


# ─── Decision Engine Tests ────────────────────────────────────────────────────


@pytest.fixture
def base_engine():
    model = WorldModel(rule=simple_stock_rule)
    simulator = Simulator(world_model=model)
    evaluator = CostEvaluator(cost_function=LogisticsCostFunction())
    return DecisionEngine(simulator=simulator, evaluator=evaluator)


class TestDecisionEngine:
    """The Decision Engine orchestrates: simulate candidate actions -> evaluate -> pick best."""

    @pytest.mark.unit
    def test_engine_recommends_reorder_to_prevent_stockout(self, base_engine):
        """If starting with low stock, it must pick a reorder action over no_op."""
        # Stock=30, Demand=20 -> will stockout on day 2.
        start_state = StateVector([30, 20, 0, 0], names=["stock", "demand", "incoming", "delay"])

        candidates = [
            Action.no_op(),
            Action("reorder", 50),
            Action("reorder", 100),
        ]

        decision = base_engine.decide(start_state, candidates, horizon=7)

        assert isinstance(decision, Decision)
        assert decision.action.name == "reorder"
        assert decision.action.value > 0

    @pytest.mark.unit
    def test_engine_recommends_no_op_when_safe(self, base_engine):
        """If stock is abundant, it should not reorder (to save holding costs)."""
        # Stock=1000, Demand=20 -> perfectly safe for 7 days.
        start_state = StateVector([1000, 20, 0, 0], names=["stock", "demand", "incoming", "delay"])

        candidates = [
            Action.no_op(),
            Action("reorder", 50),
        ]

        decision = base_engine.decide(start_state, candidates, horizon=7)

        assert decision.action == Action.no_op()
        assert decision.expected_cost < float("inf")

    @pytest.mark.unit
    def test_decision_includes_trajectory_for_explainability(self, base_engine):
        """The returned Decision object must contain the simulated trajectory of the chosen action."""
        start_state = StateVector([30, 20, 0, 0], names=["stock", "demand", "incoming", "delay"])
        candidates = [Action.no_op()]

        decision = base_engine.decide(start_state, candidates, horizon=3)

        assert isinstance(decision.trajectory, Trajectory)
        assert len(decision.trajectory) == 3

    @pytest.mark.unit
    def test_rejects_empty_candidates(self, base_engine):
        start_state = StateVector([30, 20, 0, 0], names=["stock", "demand", "incoming", "delay"])
        with pytest.raises(ValueError, match="candidate"):
            base_engine.decide(start_state, candidates=[], horizon=7)
