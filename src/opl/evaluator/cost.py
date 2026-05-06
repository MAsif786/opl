"""
Cost Evaluator — Scores simulated futures based on domain goals.

The evaluator assigns a numeric "badness" score to a trajectory.
The Decision Engine uses this to pick the action that results in the
lowest score.

Design decisions:
- Protocol-based CostFunction allows easy swapping for different industries.
- LogisticsCostFunction heavily penalizes stockouts while applying a
  smaller holding cost for excess inventory.
"""

from __future__ import annotations

from typing import Protocol

from opl.simulator.rollout import Trajectory


class CostFunction(Protocol):
    """Protocol for domain-specific cost functions."""
    def compute(self, trajectory: Trajectory) -> float:
        ...


class LogisticsCostFunction:
    """Cost function for inventory management.

    Penalizes:
    1. Stockouts (severe penalty)
    2. Holding inventory (small per-unit cost)
    """

    def __init__(
        self,
        stockout_penalty: float = 1000.0,
        holding_cost: float = 1.0,
        warehouse_capacity: float = 500.0,
        excess_holding_multiplier: float = 3.0,
        fixed_order_fee: float = 200.0,
        bulk_discount_threshold: float = 300.0,
        safety_stock_threshold: float = 0.0,
    ):
        self.stockout_penalty = stockout_penalty
        self.holding_cost = holding_cost
        self.warehouse_capacity = warehouse_capacity
        self.excess_holding_multiplier = excess_holding_multiplier
        self.fixed_order_fee = fixed_order_fee
        self.bulk_discount_threshold = bulk_discount_threshold
        self.safety_stock_threshold = safety_stock_threshold

    def compute(self, trajectory: Trajectory) -> float:
        """Calculate total cost for the simulated trajectory."""
        total_cost = 0.0
        stock_series = trajectory.get_series("stock")

        # 1. Evaluate Trajectory States (Holding, Stockout, and Safety Stock)
        for stock in stock_series:
            if stock <= 0:
                # Severe penalty for total stockout
                total_cost += self.stockout_penalty
                total_cost += abs(stock) * self.stockout_penalty
            elif stock < self.safety_stock_threshold:
                # Moderate penalty for dipping into the safety buffer
                # The further below the threshold, the higher the cost
                buffer_violation = self.safety_stock_threshold - stock
                total_cost += buffer_violation * (self.stockout_penalty * 0.1)
            else:
                # Holding cost logic...
                # Holding cost with step-function for exceeding capacity
                if stock > self.warehouse_capacity:
                    standard_cost = self.warehouse_capacity * self.holding_cost
                    excess = stock - self.warehouse_capacity
                    excess_cost = excess * (self.holding_cost * self.excess_holding_multiplier)
                    total_cost += standard_cost + excess_cost
                else:
                    total_cost += stock * self.holding_cost

        # 2. Evaluate Initial Action Costs (Supplier Pricing)
        if trajectory.initial_action and trajectory.initial_action.name == "reorder" and trajectory.initial_action.value > 0:
            order_amount = trajectory.initial_action.value
            action_cost = self.fixed_order_fee
            
            # Non-linear bulk discount step-function
            if order_amount >= self.bulk_discount_threshold:
                action_cost *= 0.5  # 50% discount on the fixed fee for bulk orders
                
            total_cost += action_cost

        return total_cost


class CostEvaluator:
    """Evaluates trajectories using a configured cost function.

    Args:
        cost_function: The domain-specific scoring function.
    """

    def __init__(self, cost_function: CostFunction) -> None:
        self.cost_function = cost_function

    def score(self, trajectory: Trajectory) -> float:
        """Assign a cost score to a trajectory (lower is better)."""
        return self.cost_function.compute(trajectory)
