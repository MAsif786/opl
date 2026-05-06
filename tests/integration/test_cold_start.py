"""
TDD Tests for Cold Start (Historical Replay).

Sprint 5 — The engine needs past prediction errors to learn from. The Cold Start
process replays historical data, generates those errors, and trains the model
so it is "smart" on Day 1 of deployment.

Tests written BEFORE implementation.
"""


import numpy as np
import pytest

from opl.engine.cold_start import ColdStart, HistoricalDay
from opl.model.action import Action
from opl.model.rules import simple_stock_rule
from opl.model.world_model import WorldModel
from opl.state.vector import StateVector


@pytest.fixture
def synthetic_history():
    """Generates a synthetic historical dataset of 90 days.
    
    The rule assumes stock decreases only by demand. 
    However, the actual history has a 5% daily spoilage rate.
    The ML correction should learn to predict this systematic bias.
    """
    np.random.seed(42)
    history = []

    stock = 1000.0
    delay = 0.0
    incoming = 0.0

    for day in range(90):
        demand = 20.0

        state = StateVector(
            [stock, demand, incoming, delay],
            names=["stock", "demand", "incoming", "delay"]
        )

        if day % 5 == 0:
            action = Action("reorder", 100)
            incoming = 100.0
            delay = 3.0
        else:
            action = Action.no_op()
            delay = max(0.0, delay - 1)
            if delay == 0:
                incoming = 0.0

        history.append(HistoricalDay(state=state, action=action))

        # Advance actual stock WITH 5% spoilage (the systematic bias)
        spoilage = stock * 0.05
        if delay == 0 and incoming > 0:
            stock = stock - demand - spoilage + incoming
        else:
            stock = stock - demand - spoilage

    return history


class TestColdStart:

    @pytest.mark.integration
    def test_replay_trains_world_model(self, synthetic_history):
        """Replaying history must result in a trained WorldModel."""
        # Note: History needs to provide S(t), Action(t), and S(t+1).
        # We pass the list of states. S(t+1) is implicitly history[t+1].state
        model = ColdStart.replay(synthetic_history, rule=simple_stock_rule)

        assert isinstance(model, WorldModel)
        assert model.correction_trained is True
        # 90 days -> 89 transitions observed
        assert len(model.error_history) == 89

    @pytest.mark.integration
    def test_replay_rejects_insufficient_history(self):
        """If we don't have enough history to train ML, it should fail early."""
        short_history = [
            HistoricalDay(
                StateVector([100, 20, 0, 0], names=["stock", "demand", "incoming", "delay"]),
                Action.no_op()
            )
        ] * 5  # Only 5 days (4 transitions) - MIN_OBSERVATIONS is 10

        with pytest.raises(ValueError, match="observations"):
            ColdStart.replay(short_history, rule=simple_stock_rule)

    @pytest.mark.slow
    @pytest.mark.integration
    def test_replay_model_beats_naive_model(self, synthetic_history):
        """A cold-started model should predict the holdout set better than a naive rule."""
        # Split 70 days train, 20 days test
        train_hist = synthetic_history[:70]
        test_hist = synthetic_history[70:]

        # 1. Train model via Cold Start replay
        trained_model = ColdStart.replay(train_hist, rule=simple_stock_rule)

        # 2. Baseline model (no training)
        naive_model = WorldModel(rule=simple_stock_rule)

        # Evaluate both on the test set
        trained_errors = []
        naive_errors = []

        for t in range(len(test_hist) - 1):
            s_t = test_hist[t].state
            action = test_hist[t].action
            s_real = test_hist[t+1].state

            # Trained prediction
            s_pred_trained = trained_model.predict(s_t, action)
            err_trained = np.abs((s_real - s_pred_trained).values).mean()
            trained_errors.append(err_trained)

            # Naive prediction
            s_pred_naive = naive_model.predict(s_t, action)
            err_naive = np.abs((s_real - s_pred_naive).values).mean()
            naive_errors.append(err_naive)

        mean_err_trained = np.mean(trained_errors)
        mean_err_naive = np.mean(naive_errors)

        # The trained model should learn the sine wave demand pattern
        # and significantly outperform the naive rule which assumes flat demand
        assert mean_err_trained < mean_err_naive
