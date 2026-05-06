import numpy as np
import pandas as pd


def generate_inventory_data(days=120, output_file="data/sample_inventory.csv"):
    """Generates a realistic daily inventory dataset for a single SKU."""
    np.random.seed(42)

    dates = pd.date_range(start="2023-01-01", periods=days, freq="D")

    data = []
    stock = 1500.0
    delay = 0.0
    incoming = 0.0

    for date in dates:
        # Realistic demand: base + weekend spike + noise
        is_weekend = date.weekday() >= 5
        demand = 40.0 + (30.0 if is_weekend else 0.0) + np.random.normal(0, 5)
        demand = max(0, demand)  # No negative demand

        # Log the beginning-of-day state
        data.append({
            "date": date.strftime("%Y-%m-%d"),
            "sku": "SKU-100",
            "stock_on_hand": stock,
            "daily_demand": demand,
            "incoming_qty": incoming,
            "lead_time_days": delay
        })

        # Policy: Reorder if stock drops below 500
        if stock < 500 and incoming == 0:
            incoming = 1000.0
            delay = 4.0  # 4 days lead time

        # Advance state
        delay = max(0, delay - 1)
        if delay == 0 and incoming > 0:
            stock = stock - demand + incoming
            incoming = 0.0
        else:
            stock = stock - demand

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Generated {output_file} with {days} days of realistic inventory data.")

if __name__ == "__main__":
    generate_inventory_data()
