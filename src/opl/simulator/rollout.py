"""
Simulator — Rolls state forward in time to evaluate actions.

This component answers the question: "What happens if we do X today?"
It takes the current state, applies an action, and then uses the WorldModel
to predict the evolution of the system over the next N days.

Design decisions:
- The proposed action is only applied on Day 1. Subsequent days assume
  a "no_op" action (we only decide for today).
- Returns a Trajectory object encapsulating the timeline, making it
  easy to score or visualize later.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from opl.model.action import Action
from opl.model.world_model import WorldModel
from opl.state.vector import StateVector


class Trajectory:
    """A sequence of states representing a simulated future.

    Args:
        states: Sequence of StateVectors representing days t+1 to t+horizon.
    """

    def __init__(self, states: Sequence[StateVector], initial_action: Action | None = None) -> None:
        if not states:
            raise ValueError("Trajectory cannot be empty")
        self.states = list(states)
        self.initial_action = initial_action

    def __len__(self) -> int:
        return len(self.states)

    def get_series(self, dimension_name: str) -> np.ndarray:
        """Extract a 1D time-series array for a specific dimension.

        Args:
            dimension_name: The name of the dimension (e.g., "stock").

        Returns:
            1D numpy array of values across the trajectory.

        Raises:
            ValueError: If the trajectory states don't have names, or the
                        name is not found.
        """
        first_state = self.states[0]
        if first_state.names is None:
            raise ValueError("Cannot get series by name; state vectors are unnamed.")

        try:
            idx = first_state.names.index(dimension_name)
        except ValueError as e:
            raise ValueError(f"Dimension '{dimension_name}' not found in state names.") from e

        return np.array([state.values[idx] for state in self.states])


class Simulator:
    """Simulates the consequences of actions over time.

    Args:
        world_model: The trained WorldModel used to predict state transitions.
    """

    def __init__(self, world_model: WorldModel) -> None:
        self.model = world_model

    def rollout(
        self,
        start_state: StateVector,
        action: Action,
        horizon: int,
    ) -> Trajectory:
        """Simulate the future by rolling the state forward.

        The specified `action` is applied on the first step.
        For all subsequent steps (day 2 to horizon), a `no_op` action is used.

        Args:
            start_state: The current state of the entity.
            action: The candidate action to evaluate today.
            horizon: Number of days to simulate forward.

        Returns:
            A Trajectory of length `horizon` containing the predicted future states.

        Raises:
            ValueError: If horizon is <= 0.
        """
        if horizon <= 0:
            raise ValueError(f"Horizon must be strictly positive, got {horizon}")

        states = []
        current_state = start_state

        # Step 1: Apply the proposed action
        next_state = self.model.predict(current_state, action)
        states.append(next_state)
        current_state = next_state

        # Step 2..N: Apply no_op for the rest of the horizon
        no_op = Action.no_op()
        for _ in range(1, horizon):
            next_state = self.model.predict(current_state, no_op)
            states.append(next_state)
            current_state = next_state

        return Trajectory(states, initial_action=action)
