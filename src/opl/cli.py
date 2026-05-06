"""
CLI Entrypoint — Runs the Decision Engine end-to-end.

This ties the whole system together:
1. Load raw data via Adapter
2. Cold Start the WorldModel on historical data
3. Setup Simulator and CostEvaluator
4. Ask DecisionEngine what to do *today*
"""

import argparse
import os
import sys

import yaml

from opl.adapter.duckdb_adapter import DuckDBAdapter
from opl.adapter.logistics import LogisticsCsvAdapter
from opl.config.schema import DomainConfig
from opl.engine.cold_start import ColdStart
from opl.engine.decision import DecisionEngine
from opl.evaluator.cost import CostEvaluator, LogisticsCostFunction
from opl.model.action import Action
from opl.model.rules import get_rule
from opl.model.world_model import WorldModel
from opl.simulator.rollout import Simulator


def main():
    parser = argparse.ArgumentParser(description="OPL Decision Engine MVP")
    parser.add_argument("--data", required=True, help="Path to historical CSV data")
    parser.add_argument("--config", required=True, help="Path to YAML domain config")
    parser.add_argument("--horizon", type=int, default=14, help="Days to simulate forward")
    parser.add_argument("--model", help="Path to save/load trained ML correction model (e.g. model.joblib)")
    parser.add_argument("--db", help="Path to DuckDB database file for logging (e.g. data/opl.db)")
    args = parser.parse_args()

    print(f"📄 Loading domain configuration from {args.config}...")
    with open(args.config) as f:
        raw_config = yaml.safe_load(f)
    config = DomainConfig(**raw_config)
    print(f"   Domain: {config.domain}")
    print(f"   Dimensions: {len(config.dimensions)}")

    print(f"📦 Loading historical data from {args.data}...")
    history = LogisticsCsvAdapter.load_history(args.data)
    print(f"   Loaded {len(history)} days of history.")

    if len(history) < 15:
        print("❌ Error: Not enough historical data to train the engine.")
        sys.exit(1)

    # Dynamically look up the physics rule from the registry
    physics_rule = get_rule(config.physics_rule)
    world_model = WorldModel(rule=physics_rule, rule_params=config.rule_params)
    loaded = False

    if args.model and os.path.exists(args.model):
        print(f"\n🧠 Loading pre-trained world model from {args.model}...")
        world_model.load(args.model)
        loaded = True

    if not loaded:
        print("\n🧠 Cold Starting the World Model...")
        # Train on all history EXCEPT the last day (which is "today")
        train_history = history[:-1]
        world_model = ColdStart.replay(train_history, rule=physics_rule)
        print(f"   ML correction trained on {len(world_model.error_history)} historical errors.")
        
        if args.model:
            print(f"💾 Saving trained model to {args.model}...")
            world_model.save(args.model)

    print("\n🔮 Booting Engine...")
    simulator = Simulator(world_model)

    # Initialize CostEvaluator dynamically using YAML params
    cost_func = LogisticsCostFunction(**config.cost_params)
    evaluator = CostEvaluator(cost_func)
    engine = DecisionEngine(simulator, evaluator)

    # "Today" is the last day in the dataset
    today = history[-1]
    current_state = today.state

    print("\n📊 Current State (Today):")
    for name, val in zip(current_state.names, current_state.values):
        print(f"   - {name}: {val:.1f}")

    # The candidate actions the operations team could take today
    candidates = [
        Action.no_op(),
        Action("reorder", 100),
        Action("reorder", 200),
        Action("reorder", 500),
        Action("reorder", 1000),
    ]

    print(f"\n⚙️  Evaluating {len(candidates)} candidate actions over a {args.horizon}-day horizon...")
    decision = engine.decide(current_state, candidates, horizon=args.horizon)

    print("\n==================================================")
    print("🎯 RECOMMENDED DECISION")
    print("==================================================")
    print(f"Action to take: {decision.action.name.upper()} {decision.action.value}")
    print(f"Expected Cost:  {decision.expected_cost:.2f}")
    print("--------------------------------------------------")
    print("Future Trajectory (Next 5 Days):")

    stock_series = decision.trajectory.get_series("stock")
    demand_series = decision.trajectory.get_series("demand")

    for i in range(min(5, args.horizon)):
        print(f"  Day {i+1}: Stock = {stock_series[i]:.1f} | Expected Demand = {demand_series[i]:.1f}")

    if decision.action.name == "reorder" and decision.action.value > 0:
        print("  ... (Arrival expected on Day 4)")

    print("==================================================\n")

    # --- SQL Logging ---
    if args.db:
        print(f"💾 Logging observation to DuckDB ({args.db})...")
        db = DuckDBAdapter(args.db)
        # We don't know the 'reality' of tomorrow yet, so we log the state and action taken.
        # In a real system, a separate process would fill in the 'next_state' tomorrow.
        db.log_observation(
            entity_id=config.domain, 
            state=current_state, 
            action=decision.action, 
            next_state=decision.trajectory.states[0] # The engine's best guess for tomorrow
        )
        db.close()
        print("   Observation logged successfully.")


if __name__ == "__main__":
    main()
