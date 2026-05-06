"""
Tests for DuckDBAdapter.
"""

import os

import pytest

from opl.adapter.duckdb_adapter import DuckDBAdapter
from opl.model.action import Action
from opl.state.vector import StateVector


class TestDuckDBAdapter:
    @pytest.fixture
    def adapter(self, tmp_path):
        db_file = os.path.join(tmp_path, "test_opl.db")
        adapter = DuckDBAdapter(db_file)
        yield adapter
        adapter.close()

    @pytest.mark.unit
    def test_log_and_load_observation(self, adapter):
        """Verify that we can save a state and reload it exactly."""
        s1 = StateVector([100, 20, 0, 0], names=["stock", "demand", "incoming", "delay"])
        a1 = Action("reorder", 500)
        s2 = StateVector([80, 20, 500, 3], names=["stock", "demand", "incoming", "delay"])

        adapter.log_observation("SKU-TEST", s1, a1, s2)

        history = adapter.load_history("SKU-TEST")

        assert len(history) == 1
        assert history[0].state == s1
        assert history[0].action == a1
        assert history[0].state.names == ("stock", "demand", "incoming", "delay")

    @pytest.mark.unit
    def test_multiple_entities_isolation(self, adapter):
        """Verify that loading history for one entity doesn't return data for another."""
        s1 = StateVector([100], names=["stock"])
        a1 = Action.no_op()

        adapter.log_observation("SKU-A", s1, a1, s1)
        adapter.log_observation("SKU-B", s1, a1, s1)

        history_a = adapter.load_history("SKU-A")
        assert len(history_a) == 1

        history_c = adapter.load_history("SKU-C")
        assert len(history_c) == 0
