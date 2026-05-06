from __future__ import annotations

import duckdb
import numpy as np
from collections.abc import Sequence

from opl.engine.cold_start import HistoricalDay
from opl.model.action import Action
from opl.state.vector import StateVector


class DuckDBAdapter:
    """SQL-based data adapter using DuckDB with Incremental Training support."""

    def __init__(self, db_path: str) -> None:
        self.conn = duckdb.connect(db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        """Create the necessary tables if they don't exist."""
        # Reality: What actually happened (State -> Action -> Next State)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS observations (
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                entity_id VARCHAR,
                dimension_names JSON,
                state_values DOUBLE[],
                action_name VARCHAR,
                action_value DOUBLE,
                next_state_values DOUBLE[],
                is_trained BOOLEAN DEFAULT FALSE
            )
        """)

        # Intelligence: What the engine RECOMMENDED (State -> Choice -> Score)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS decisions (
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                entity_id VARCHAR,
                current_state_values DOUBLE[],
                recommended_action_name VARCHAR,
                recommended_action_value DOUBLE,
                score DOUBLE,
                horizon_days INTEGER
            )
        """)

    def log_decision(
        self, entity_id: str, state: StateVector, action: Action, score: float, horizon: int
    ) -> None:
        """Log a recommendation made by the engine."""
        self.conn.execute(
            """
            INSERT INTO decisions 
            (entity_id, current_state_values, recommended_action_name, recommended_action_value, score, horizon_days) 
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [entity_id, state.values.tolist(), action.name, action.value, score, horizon],
        )

    def log_observation(self, entity_id: str, state: StateVector, action: Action, next_state: StateVector) -> None:
        """Save a real-world transition. Defaults to is_trained=FALSE."""
        import json

        self.conn.execute(
            """
            INSERT INTO observations
            (entity_id, dimension_names, state_values, action_name, action_value, next_state_values, is_trained)
            VALUES (?, ?, ?, ?, ?, ?, FALSE)
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

    def load_history(self, entity_id: str, only_new: bool = True) -> Sequence[HistoricalDay]:
        """Load history for a specific entity."""
        where = "WHERE entity_id = ?"
        if only_new:
            where += " AND is_trained = FALSE"
        return self._fetch_history(where, [entity_id])

    def load_all_history(self, only_new: bool = True) -> Sequence[HistoricalDay]:
        """Load historical observations from the entire database."""
        where = "WHERE is_trained = FALSE" if only_new else ""
        return self._fetch_history(where, [])

    def mark_as_trained(self, entity_id: str | None = None) -> None:
        """Mark processed records as trained so they aren't used again."""
        if entity_id:
            self.conn.execute("UPDATE observations SET is_trained = TRUE WHERE entity_id = ?", [entity_id])
        else:
            self.conn.execute("UPDATE observations SET is_trained = TRUE WHERE is_trained = FALSE")

    def _fetch_history(self, where_clause: str, params: list) -> Sequence[HistoricalDay]:
        import json

        query = f"""
            SELECT dimension_names, state_values, action_name, action_value 
            FROM observations 
            {where_clause}
            ORDER BY timestamp ASC
        """
        res = self.conn.execute(query, params).fetchall()

        history = []
        for row in res:
            names = json.loads(row[0])
            state = StateVector(np.array(row[1]), names=names)
            action = Action(row[2], row[3])
            history.append(HistoricalDay(state=state, action=action))
        return history

    def close(self) -> None:
        self.conn.close()
