"""
DuckDB Adapter — Persistent SQL storage for state observations and history.

This adapter allows the engine to:
1. Log real-world observations into a structured SQL database.
2. Query historical data efficiently for ML retraining.
3. Handle large-scale datasets without loading everything into Python memory.
"""

from __future__ import annotations

from collections.abc import Sequence

import duckdb
import numpy as np

from opl.engine.cold_start import HistoricalDay
from opl.model.action import Action
from opl.state.vector import StateVector


class DuckDBAdapter:
    """SQL-based data adapter using DuckDB.

    Args:
        db_path: Path to the DuckDB file (e.g., 'data/opl.db').
    """

    def __init__(self, db_path: str) -> None:
        self.conn = duckdb.connect(db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        """Create the necessary tables if they don't exist."""
        # We store everything in a single wide table for simplicity in the MVP.
        # entity_id allows tracking multiple SKUs/Warehouses in one DB.
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS observations (
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                entity_id VARCHAR,
                dimension_names JSON,
                state_values DOUBLE[],
                action_name VARCHAR,
                action_value DOUBLE,
                next_state_values DOUBLE[]
            )
        """)

    def log_observation(self, entity_id: str, state: StateVector, action: Action, next_state: StateVector) -> None:
        """Save a real-world transition to the database.

        Args:
            entity_id: Unique ID of the SKU or warehouse.
            state: State before the action.
            action: Action taken.
            next_state: Resulting state observed in reality.
        """
        import json

        self.conn.execute(
            """
            INSERT INTO observations
            (entity_id, dimension_names, state_values, action_name, action_value, next_state_values)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                entity_id,
                json.dumps(state.names),
                state.values.tolist(),
                action.name,
                action.value,
                next_state.values.tolist(),
            ],
        )

    def load_history(self, entity_id: str) -> Sequence[HistoricalDay]:
        """Load historical observations for a specific entity from SQL.

        Returns:
            A sequence of HistoricalDay objects for the ColdStart engine.
        """
        import json

        res = self.conn.execute(
            """
            SELECT dimension_names, state_values, action_name, action_value
            FROM observations
            WHERE entity_id = ?
            ORDER BY timestamp ASC
            """,
            [entity_id],
        ).fetchall()

        history = []
        for row in res:
            names = json.loads(row[0])
            state = StateVector(np.array(row[1]), names=names)
            action = Action(row[2], row[3])
            history.append(HistoricalDay(state=state, action=action))

        return history

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()
