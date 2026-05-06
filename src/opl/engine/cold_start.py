"""
Cold Start — Historical Replay.

Before deploying the engine, we need the ML correction layer to be trained.
Otherwise, it starts with zero understanding of the system's actual behavior.

The ColdStart class replays historical data day-by-day. It observes the
state, predicts the next state using the base rule, compares it to what
actually happened (reality), records the error, and trains the model.

Design decisions:
- Replay is strictly chronological to prevent data leakage.
- Output is a fully trained WorldModel ready for Day 1 production use.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from opl.model.action import Action
from opl.model.world_model import RuleFunction, WorldModel
from opl.state.vector import StateVector


@dataclass(frozen=True)
class HistoricalDay:
    """A snapshot of a single historical day.

    Args:
        state: The actual state vector recorded on that day.
        action: The actual action taken on that day.
    """

    state: StateVector
    action: Action


class ColdStart:
    """Replays historical data to train a WorldModel's correction layer."""

    @staticmethod
    def replay(
        history: Sequence[HistoricalDay],
        rule: RuleFunction,
        alpha: float = 1.0,
    ) -> WorldModel:
        """Walk through history, record errors, and train the model.

        Args:
            history: Chronological list of historical days.
            rule: The base physics rule for the world model.
            alpha: Regularization strength for the ML model.

        Returns:
            A WorldModel with a trained ML correction layer.

        Raises:
            ValueError: If the history is too short to train the model.
        """
        # Need at least N days to get N-1 transitions
        # WorldModel needs MIN_OBSERVATIONS (10).
        # So we need at least MIN_OBSERVATIONS + 1 days of history.
        if len(history) < 2:
            raise ValueError("History too short for replay.")

        model = WorldModel(rule=rule)

        # For each day t, we try to predict t+1. Then we look at the real t+1
        # to record the error.
        for t in range(len(history) - 1):
            s_t = history[t].state
            action = history[t].action
            s_real = history[t + 1].state

            # Record observation (predicts using rule, compares to reality, logs error)
            model.record_observation(s_t, action, s_real)

        # After walking through all history, train the correction model
        model.train_correction(alpha=alpha)

        return model
