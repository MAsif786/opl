"""
TDD Tests for Phase 2: Multi-Entity Systems.

These tests prove Section 23 of the PRD: that we can evolve from single-entity
to system-level decisions simply by expanding the StateVector and Action space,
without changing the core WorldModel or Simulator architectures.

Tests written BEFORE implementation.
"""

import numpy as np
import pytest

from opl.model.action import Action
from opl.model.rules import multi_warehouse_rule
from opl.model.world_model import WorldModel
from opl.state.vector import StateVector


class TestMultiEntity:
    """Testing multi-warehouse interactions."""

    @pytest.fixture
    def multi_state(self):
        """
        State schema:
        0: local_stock
        1: demand
        2: incoming (from supplier or transfer)
        3: delay
        4: other_wh_stock (Warehouse B)
        5: transport_time (Days to transfer B -> A)
        """
        return StateVector(
            [10, 20, 0, 0, 500, 2],
            names=["local_stock", "demand", "incoming", "delay", "other_wh_stock", "transport_time"],
        )

    @pytest.mark.unit
    def test_multi_warehouse_rule_no_op(self, multi_state):
        """Without action, both warehouses evolve normally."""
        # Action is no_op. Local stock decreases by demand. Other WH stock remains (for simplicity).
        action = Action.no_op()
        s_next = multi_warehouse_rule(multi_state, action)

        assert s_next[0] == -10.0  # local_stock = 10 - 20
        assert s_next[4] == 500.0  # other_wh_stock unchanged

    @pytest.mark.unit
    def test_transfer_action_mechanics(self, multi_state):
        """A transfer action takes stock from WH_B and puts it in transit to WH_A."""
        action = Action("transfer", 100)
        s_next = multi_warehouse_rule(multi_state, action)

        # WH_B stock drops immediately
        assert s_next[4] == 400.0  # 500 - 100
        # Transit is set up
        assert s_next[2] == 100.0  # incoming
        assert s_next[3] == 2.0  # delay = transport_time

    @pytest.mark.unit
    def test_transfer_fails_if_other_wh_empty(self, multi_state):
        """Cannot transfer more than WH_B has."""
        empty_b_state = StateVector(
            [10, 20, 0, 0, 50, 2],
            names=["local_stock", "demand", "incoming", "delay", "other_wh_stock", "transport_time"],
        )
        action = Action("transfer", 100)
        s_next = multi_warehouse_rule(empty_b_state, action)

        # Should cap transfer at 50
        assert s_next[4] == 0.0  # WH_B empty
        assert s_next[2] == 50.0  # Only 50 incoming

    @pytest.mark.unit
    def test_world_model_learns_cross_entity_interactions(self):
        """
        Prove that the core ML engine learns complex interactions automatically.
        Suppose WH_B secretly reserves 20% of its stock and won't actually send it.
        The base rule doesn't know this, but the ML correction should learn it.
        """
        model = WorldModel(rule=multi_warehouse_rule)
        np.random.seed(42)

        for _ in range(50):
            other_stock = float(np.random.randint(100, 1000))
            transfer_qty = float(np.random.randint(50, 200))

            s_t = StateVector(
                [10, 20, 0, 0, other_stock, 2],
                names=["local_stock", "demand", "incoming", "delay", "other_wh_stock", "transport_time"],
            )
            action = Action("transfer", transfer_qty)

            # The rule will predict incoming = transfer_qty.
            # But REALITY: WH_B only sent 80% of what was requested due to internal reserves.
            real_sent = transfer_qty * 0.8

            s_real = StateVector(
                [-10, 20, real_sent, 2, other_stock - real_sent, 2],
                names=["local_stock", "demand", "incoming", "delay", "other_wh_stock", "transport_time"],
            )
            model.record_observation(s_t, action, s_real)

        model.train_correction()

        # Test prediction
        test_state = StateVector(
            [10, 20, 0, 0, 500, 2],
            names=["local_stock", "demand", "incoming", "delay", "other_wh_stock", "transport_time"],
        )
        test_action = Action("transfer", 100)

        # Rule predicts incoming = 100
        # Reality should be incoming = 80
        # Let's see if the model learned to predict closer to 80
        s_pred = model.predict(test_state, test_action)

        # The ML model should correct the `incoming` dimension (index 2) downwards
        assert s_pred[2] < 95.0
        assert s_pred[2] == pytest.approx(80.0, rel=0.15)
