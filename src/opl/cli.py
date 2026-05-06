from __future__ import annotations

import argparse
import sys
from pathlib import Path

from opl.adapter.duckdb_adapter import DuckDBAdapter
from opl.adapter.logistics import LogisticsCsvAdapter
from opl.config import load_config
from opl.engine.decision import DecisionEngine
from opl.model.action import Action
from opl.model.rules import get_rule
from opl.model.world_model import WorldModel
from opl.state.builder import StateBuilder


def main():
    parser = argparse.ArgumentParser(description="OPL Cognitive Decision Engine")
    parser.add_argument("--config", type=str, default="data/domain_logistics.yaml", help="Path to domain config")
    parser.add_argument("--db", type=str, default="data/opl_audit.db", help="Path to DuckDB audit store")
    parser.add_argument("--model-path", type=str, default="data/world_model.joblib", help="Path to save/load model")

    subparsers = parser.add_subparsers(dest="command", help="Subcommand to run")

    # Train Command (Dual Mode)
    train_parser = subparsers.add_parser("train", help="Train the hybrid correction model")
    train_parser.add_argument(
        "--mode", 
        choices=["cold-start", "retrain"], 
        required=True, 
        help="cold-start (from raw CSV) or retrain (from SQL audit log)"
    )
    train_parser.add_argument("--data", type=str, help="Path to raw CSV (required for cold-start)")
    train_parser.add_argument("--entity", type=str, help="Entity ID (optional for retrain, defaults to all)")

    # Predict Command
    predict_parser = subparsers.add_parser("predict", help="Recommend the best action for an entity")
    predict_parser.add_argument("--entity", type=str, required=True, help="Entity ID (e.g. SKU_WH_1)")
    predict_parser.add_argument("--horizon", type=int, default=7, help="Simulation horizon")

    # Simulate Command
    sim_parser = subparsers.add_parser("simulate", help="Simulate a specific action rollout")
    sim_parser.add_argument("--entity", type=str, required=True, help="Entity ID")
    sim_parser.add_argument("--action", type=str, required=True, help="Action name")
    sim_parser.add_argument("--value", type=float, required=True, help="Action value")
    sim_parser.add_argument("--horizon", type=int, default=14, help="Simulation horizon")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Setup Infrastructure
    config = load_config(args.config)
    builder = StateBuilder(config)
    db = DuckDBAdapter(args.db)
    rule = get_rule(config.physics_rule)

    model = WorldModel(rule=rule, params=config.rule_params)
    if Path(args.model_path).exists():
        model.load(args.model_path)

    if args.command == "train":
        history = []
        if args.mode == "cold-start":
            if not args.data:
                print("[!] Error: --data <csv_path> is required for cold-start mode.")
                sys.exit(1)
            print(f"[*] Cold-starting from raw data: {args.data}...")
            history = LogisticsCsvAdapter.load_history(args.data)
        
        elif args.mode == "retrain":
            print(f"[*] Retraining from SQL audit trail ({args.db})...")
            # If entity is None, the adapter should ideally support 'load_all', 
            # but we'll use a sample SKU for this demonstration.
            history = db.load_history(args.entity or "SKU_WH_1")

        if len(history) < 10:
            print(f"[!] Not enough history to train (found {len(history)}). Need 10+ observations.")
            sys.exit(1)

        # Populate WorldModel error log by replaying history
        for i in range(len(history) - 1):
            curr = history[i]
            next_actual = history[i+1].state
            model.record_observation(curr.state, curr.action, next_actual)

        model.train_correction()
        model.save(args.model_path)
        print(f"[+] Model successfully trained and saved to {args.model_path}")

    elif args.command == "predict":
        engine = DecisionEngine(model, config)
        history = db.load_history(args.entity)
        if not history:
            print(f"[!] No current state found for {args.entity}. Run cold-start or simulate first.")
            sys.exit(1)

        current_state = history[-1].state
        print(f"[*] Current State for {args.entity}: {current_state.to_dict()}")

        candidates = [Action.no_op(), Action("reorder", 50), Action("reorder(100)", 100)]
        decision = engine.decide(current_state, candidates, horizon=args.horizon)

        print("\n--- OPTIMIZED DECISION ---")
        print(f"Action: {decision.action}")
        print(f"Score:  {decision.score:.2f}")
        print("\n--- PROJECTED TRAJECTORY ---")
        print(decision.trajectory.as_dataframe().head(5))

    elif args.command == "simulate":
        from opl.simulator.rollout import rollout
        
        # Try to find last state in DB, otherwise use a default
        history = db.load_history(args.entity)
        if history:
            start_state = history[-1].state
        else:
            print(f"[!] No history for {args.entity}, using dummy initial state.")
            start_state = builder.build({"stock_on_hand": 100, "daily_demand": 20, "incoming_qty": 0, "lead_time_days": 3})

        action = Action(args.action, args.value)
        trajectory = rollout(model, start_state, action, horizon=args.horizon)
        
        print(f"[*] Simulation Result for {args.entity}:")
        print(trajectory.as_dataframe())
        
        # Audit the first step of this simulation as an observation
        db.log_observation(args.entity, start_state, action, trajectory.states[0])


if __name__ == "__main__":
    main()
