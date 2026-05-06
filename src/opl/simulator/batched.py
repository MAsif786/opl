"""
Batched Simulator for enterprise scale (100k+ entities).

While `Simulator` handles single entities for deep explainability and debugging,
`BatchedSimulator` processes massive 2D NumPy matrices. It allows the engine
to evaluate an action across 100,000 SKUs simultaneously in milliseconds.
"""

import numpy as np

from opl.model.action import Action
from opl.model.world_model import WorldModel


class BatchedSimulator:
    """Simulates N entities concurrently using 2D matrices."""

    def __init__(self, world_model: WorldModel) -> None:
        # Note: In a fully complete system, the world_model would also need
        # to expose a `predict_batch(matrix)` method. For this MVP, we
        # bypass the ML layer for batched simulation and use the pure physics rule.
        self.world_model = world_model

    def rollout_batch(
        self, start_matrix: np.ndarray, action: Action, horizon: int
    ) -> list[np.ndarray]:
        """Roll a batch of entities forward in time.

        Args:
            start_matrix: 2D numpy array of shape (batch_size, num_dimensions).
            action: The candidate action applied on Day 1.
            horizon: Number of days to simulate.

        Returns:
            A list of 2D numpy arrays, representing the state matrix at each day.
        """
        if horizon <= 0:
            raise ValueError(f"Horizon must be strictly positive, got {horizon}")

        states = [start_matrix]
        current_matrix = start_matrix

        # Day 1: Apply Action
        current_matrix = self._apply_vectorized_rule(current_matrix, action)
        states.append(current_matrix)

        # Day 2..N: Apply No-Op
        no_op = Action.no_op()
        for _ in range(1, horizon):
            current_matrix = self._apply_vectorized_rule(current_matrix, no_op)
            states.append(current_matrix)

        return states

    @staticmethod
    def _apply_vectorized_rule(s_matrix: np.ndarray, action: Action) -> np.ndarray:
        """Vectorized version of simple_stock_rule."""
        next_s = np.copy(s_matrix)

        stock = next_s[:, 0]
        demand = next_s[:, 1]
        incoming = next_s[:, 2]
        delay = next_s[:, 3]

        arrivals = (delay <= 0) & (incoming > 0)

        stock[arrivals] = stock[arrivals] - demand[arrivals] + incoming[arrivals]
        stock[~arrivals] = stock[~arrivals] - demand[~arrivals]

        incoming[arrivals] = 0.0
        delay[arrivals] = 0.0
        delay[~arrivals] = np.maximum(0.0, delay[~arrivals] - 1)

        if action.name == "reorder" and action.value > 0:
            incoming[:] = action.value
            delay[:] = 3.0

        return next_s
