"""
TDD Tests for Vectorized Scale (100k Entities).

Proves that the engine can simulate 100,000 SKUs concurrently in under 1 minute
by utilizing 2D NumPy matrices.
"""

import time

import numpy as np
import pytest

from opl.model.action import Action
from opl.model.rules import simple_stock_rule
from opl.model.world_model import WorldModel
from opl.simulator.batched import BatchedSimulator
from opl.state.vector import StateVector


class TestVectorizedScale:
    @pytest.mark.unit
    def test_simulate_100k_skus(self):
        """Prove 100,000 entities can be rolled forward 14 days very quickly."""
        batch_size = 100_000
        horizon = 14

        # Generate random initial states for 100k SKUs
        # [stock, demand, incoming, delay]
        stock_col = np.random.uniform(50, 1000, size=(batch_size, 1))
        demand_col = np.random.uniform(5, 50, size=(batch_size, 1))
        incoming_col = np.zeros((batch_size, 1))
        delay_col = np.zeros((batch_size, 1))

        initial_matrix = np.hstack([stock_col, demand_col, incoming_col, delay_col])

        # Initialize Simulator
        # Dummy world model for the MVP batched implementation
        dummy_model = WorldModel(rule=simple_stock_rule)
        simulator = BatchedSimulator(dummy_model)

        # Roll forward
        start_time = time.time()

        action = Action("reorder", 100)
        states_over_time = simulator.rollout_batch(initial_matrix, action, horizon=horizon)

        end_time = time.time()
        duration = end_time - start_time

        # 14 days * 100k SKUs = 1.4 million state transitions
        assert len(states_over_time) == 15  # Initial + 14 days
        assert states_over_time[-1].shape == (batch_size, 4)

        # Should take significantly less than 1 second on CPU
        assert duration < 1.0, f"Took too long: {duration} seconds"
        print(f"\n🚀 Simulated 100,000 SKUs for {horizon} days in {duration:.4f} seconds!")

    @pytest.mark.unit
    def test_vectorized_math_is_correct(self):
        """Prove that the 2D vectorized rule outputs exactly the same as the 1D scalar rule."""
        # Create a small batch of specific test cases
        # Row 0: Normal day (stock 100, demand 20, no incoming)
        # Row 1: Arrival day (stock 50, demand 10, incoming 100, delay 0)
        # Row 2: In transit (stock 200, demand 30, incoming 50, delay 2)
        initial_matrix = np.array([[100.0, 20.0, 0.0, 0.0], [50.0, 10.0, 100.0, 0.0], [200.0, 30.0, 50.0, 2.0]])

        # 1. Test NO_OP Action
        action_noop = Action.no_op()
        vectorized_noop = BatchedSimulator._apply_vectorized_rule(initial_matrix, action_noop)

        # Verify Row 0 scalar
        s0 = StateVector(initial_matrix[0].tolist(), names=["stock", "demand", "incoming", "delay"])
        scalar0 = simple_stock_rule(s0, action_noop)
        np.testing.assert_array_almost_equal(vectorized_noop[0], scalar0.values)

        # Verify Row 1 scalar (arrival logic)
        s1 = StateVector(initial_matrix[1].tolist(), names=["stock", "demand", "incoming", "delay"])
        scalar1 = simple_stock_rule(s1, action_noop)
        np.testing.assert_array_almost_equal(vectorized_noop[1], scalar1.values)

        # Verify Row 2 scalar (delay decrement)
        s2 = StateVector(initial_matrix[2].tolist(), names=["stock", "demand", "incoming", "delay"])
        scalar2 = simple_stock_rule(s2, action_noop)
        np.testing.assert_array_almost_equal(vectorized_noop[2], scalar2.values)

        # 2. Test REORDER Action
        action_reorder = Action("reorder", 300)
        vectorized_reorder = BatchedSimulator._apply_vectorized_rule(initial_matrix, action_reorder)

        # Row 0 should trigger a reorder (incoming=300, delay=3)
        scalar0_reorder = simple_stock_rule(s0, action_reorder)
        np.testing.assert_array_almost_equal(vectorized_reorder[0], scalar0_reorder.values)

        # Row 2 already has incoming, so standard reorder is ignored by simple_stock_rule
        scalar2_reorder = simple_stock_rule(s2, action_reorder)
        np.testing.assert_array_almost_equal(vectorized_reorder[2], scalar2_reorder.values)
