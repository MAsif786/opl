"""
Decision Engine — The orchestrator.

The Decision Engine ties everything together. Given a current state
and a list of candidate actions, it:
1. Simulates the future for each action (via Simulator)
2. Evaluates each simulated future (via CostEvaluator)
3. Returns the action that minimizes cost.

Design decisions:
- Decision dataclass wraps the best action + its simulated trajectory
  and expected cost. This makes the system fully explainable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from opl.evaluator.cost import CostEvaluator
from opl.model.action import Action
from opl.simulator.rollout import Simulator, Trajectory
from opl.state.vector import StateVector


@dataclass(frozen=True)
class Decision:
    """The result of the decision process.

    Args:
        action: The recommended action to take today.
        expected_cost: The computed cost of the simulated future.
        trajectory: The predicted states for the next N days.
    """
    action: Action
    expected_cost: float
    trajectory: Trajectory


class DecisionEngine:
    """Orchestrates simulation and evaluation to pick the best action.

    Args:
        simulator: Engine to roll states forward.
        evaluator: Scorer to evaluate trajectories.
    """

    def __init__(self, simulator: Simulator, evaluator: CostEvaluator) -> None:
        self.simulator = simulator
        self.evaluator = evaluator

    def decide(
        self,
        current_state: StateVector,
        candidates: Sequence[Action],
        horizon: int = 14,
    ) -> Decision:
        """Evaluate candidates and return the best decision.

        Args:
            current_state: State snapshot of the entity today.
            candidates: List of actions to test.
            horizon: Days to look ahead in the simulation.

        Returns:
            Decision object containing the best action and its expected trajectory.

        Raises:
            ValueError: If candidates list is empty.
        """
        if not candidates:
            raise ValueError("Must provide at least one candidate action.")

        best_decision = None
        min_cost = float("inf")

        for action in candidates:
            # 1. Simulate what happens if we take this action
            trajectory = self.simulator.rollout(current_state, action, horizon)

            # 2. Score the resulting future
            cost = self.evaluator.score(trajectory)

            # 3. Keep the best (lowest cost)
            if cost < min_cost:
                min_cost = cost
                best_decision = Decision(
                    action=action,
                    expected_cost=cost,
                    trajectory=trajectory,
                )

        assert best_decision is not None
        return best_decision
