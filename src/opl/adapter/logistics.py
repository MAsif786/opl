"""
Logistics Adapter — Maps raw industry data to engine formats.

In a real deployment, this is the ONLY layer that changes per customer
or industry. It reads raw data (CSV, DB, etc.) and formats it into
HistoricalDays that the engine understands.
"""

import pandas as pd

from opl.engine.cold_start import HistoricalDay
from opl.model.action import Action
from opl.state.vector import StateVector


class LogisticsCsvAdapter:
    """Reads logistics inventory CSVs and builds engine history."""

    @staticmethod
    def load_history(filepath: str) -> list[HistoricalDay]:
        """Load history from a CSV file.
        
        Expected CSV columns:
        date, sku, stock_on_hand, daily_demand, incoming_qty, lead_time_days
        
        Assumes action was 'reorder N' if incoming_qty > 0 on a given day
        AND it was 0 the day before. (Simplified for MVP).
        """
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        history = []

        for idx, row in df.iterrows():
            # Build state vector
            state = StateVector(
                [
                    float(row['stock_on_hand']),
                    float(row['daily_demand']),
                    float(row['incoming_qty']),
                    float(row['lead_time_days'])
                ],
                names=["stock", "demand", "incoming", "delay"]
            )

            # Infer action taken on this day.
            # In real life, this comes from a "decisions" or "PO" table.
            # Here, if incoming_qty > 0 and was 0 yesterday, a reorder happened.
            action = Action.no_op()
            if idx > 0 and row['incoming_qty'] > 0 and df.iloc[idx-1]['incoming_qty'] == 0:
                action = Action("reorder", float(row['incoming_qty']))

            history.append(HistoricalDay(state=state, action=action))

        return history
