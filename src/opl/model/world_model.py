"""
WorldModel — Hybrid prediction engine (rule + ML correction).

This is the cognitive core of the system. It predicts how state evolves:

    S_pred(t+1) = rule(S(t), action) + correction(S(t), action)

The rule encodes domain physics. The correction learns systematic errors
from past observations. Together, they adapt to reality.

Design decisions:
- Rule is a pure function injected at construction time
- Correction is optional — off until trained
- Error history stored as structured records for training
- ML correction predicts the ERROR, not the state (faster convergence)
- Minimum observation threshold prevents overfitting on tiny samples
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.linear_model import Ridge

from opl.model.action import Action
from opl.state.vector import StateVector

# Minimum observations before ML correction can be trained
MIN_OBSERVATIONS = 10

# Protocol for physics rules: (State, Action, Params) -> State
RuleFunction = Callable[[StateVector, Action, dict | None], StateVector]


@dataclass
class ObservationRecord:
    """A single observed transition: state + action → real next state."""

    s_t: StateVector
    action: Action
    s_predicted: StateVector
    s_real: StateVector
    error: StateVector  # real - predicted


class WorldModel:
    """Hybrid world model: deterministic rule + learned ML correction.

    Args:
        rule: A function (StateVector, Action) → StateVector encoding domain physics.
    """

    def __init__(
        self,
        rule: RuleFunction,
        rule_params: dict | None = None,
    ) -> None:
        self._rule = rule
        self._rule_params = rule_params or {}
        self._error_history: list[ObservationRecord] = []
        self._correction_model: Ridge | None = None
        self._correction_trained: bool = False

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def error_history(self) -> list[ObservationRecord]:
        """Recorded observation history."""
        return list(self._error_history)

    @property
    def correction_trained(self) -> bool:
        """Whether the ML correction layer has been trained."""
        return self._correction_trained

    # ── Prediction ────────────────────────────────────────────────────────

    def predict(self, s_t: StateVector, action: Action) -> StateVector:
        """Predict the next state given current state and action.

        If ML correction is trained, applies correction to the rule output.

        Args:
            s_t: Current state.
            action: Action being taken.

        Returns:
            Predicted next state.
        """
        s_rule = self._rule(s_t, action, self._rule_params)

        if not self._correction_trained or self._correction_model is None:
            return s_rule

        # Build feature vector for correction model
        features = self._build_features(s_t, action)
        correction = self._correction_model.predict(features.reshape(1, -1))[0]

        # Apply correction: predicted = rule + learned_error_correction
        corrected_values = s_rule.values + correction
        return StateVector(corrected_values, names=s_rule.names)

    # ── Error Computation ─────────────────────────────────────────────────

    @staticmethod
    def compute_error(predicted: StateVector, real: StateVector) -> StateVector:
        """Compute prediction error: real - predicted.

        Args:
            predicted: What the model predicted.
            real: What actually happened.

        Returns:
            Error vector (real - predicted).
        """
        return real - predicted

    # ── Observation Recording ─────────────────────────────────────────────

    def record_observation(
        self,
        s_t: StateVector,
        action: Action,
        s_real: StateVector,
    ) -> ObservationRecord:
        """Record a real-world observation for learning.

        Predicts using current model, computes error, stores for training.

        Args:
            s_t: State at time t.
            action: Action taken at time t.
            s_real: Real observed state at time t+1.

        Returns:
            The recorded observation.
        """
        s_predicted = self._rule(s_t, action, self._rule_params)
        error = self.compute_error(s_predicted, s_real)

        record = ObservationRecord(
            s_t=s_t,
            action=action,
            s_predicted=s_predicted,
            s_real=s_real,
            error=error,
        )
        self._error_history.append(record)
        return record

    # ── ML Correction Training ────────────────────────────────────────────

    def train_correction(self, alpha: float = 1.0) -> None:
        """Train the ML correction model on recorded error history.

        The correction model learns to predict the systematic error
        of the rule-based prediction, given the input state and action.

        Args:
            alpha: Ridge regularization strength.

        Raises:
            ValueError: If insufficient observations for training.
        """
        n = len(self._error_history)
        if n < MIN_OBSERVATIONS:
            raise ValueError(
                f"Need at least {MIN_OBSERVATIONS} observations to train "
                f"correction, got {n}"
            )

        # Build training data
        X = np.array([
            self._build_features(rec.s_t, rec.action)
            for rec in self._error_history
        ])
        # Target: the error the rule made (we learn to predict it)
        Y = np.array([rec.error.values for rec in self._error_history])

        # Train Ridge regression (fast, stable, CPU-friendly)
        self._correction_model = Ridge(alpha=alpha)
        self._correction_model.fit(X, Y)
        self._correction_trained = True

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save the trained correction model to disk.

        Args:
            path: File path (e.g., 'model.joblib').

        Raises:
            ValueError: If the model hasn't been trained yet.
        """
        if not self._correction_trained or self._correction_model is None:
            raise ValueError("Cannot save an untrained model.")
        
        joblib.dump(self._correction_model, path)

    def load(self, path: str) -> None:
        """Load a trained correction model from disk.

        Args:
            path: File path to a joblib-serialized model.
        """
        self._correction_model = joblib.load(path)
        self._correction_trained = True

    # ── Feature Engineering ───────────────────────────────────────────────

    @staticmethod
    def _build_features(s_t: StateVector, action: Action) -> np.ndarray:
        """Build feature vector for the ML correction model.

        Concatenates state values with action value.

        Args:
            s_t: Current state.
            action: Action being taken.

        Returns:
            1D numpy array of features.
        """
        return np.concatenate([s_t.values, [action.value]])
