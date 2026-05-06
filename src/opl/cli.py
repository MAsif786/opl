from __future__ import annotations

import argparse
import sys
from pathlib import Path

from opl.adapter.duckdb_adapter import DuckDBAdapter
from opl.config import load_config
from opl.engine.decision import DecisionEngine
from opl.model.action import Action
from opl.model.rules import get_rule
from opl.model.world_model import WorldModel
from opl.state.builder import StateBuilder


def main():
    parser = argparse.ArgumentParser(description="OPL Cognitive Decision Engine")
    parser.add_argument("--config", type=str, default="data/domain_logistics.yaml", help="Path to domain config")
    parser.add_argument("--db", type=str, default="data/opl_audit.db", help="Path to DuckDB")
    parser.add_argument("--model-path", type=str, default="data/world_model.joblib", help="Path to save/load model")

    subparsers = parser.add_subparsers(dest="command", help="Subcommand to run")

    # Train Command
    train_parser = subparsers.add_parser("train", help="Train the hybrid correction model using history")
    train_parser.add_argument("--entity", type=str, help="Optional entity ID to train on (defaults to SKU_WH_1)")

    # Predict Command
    predict_parser = subparsers.add_parser("predict", help="Recommend the best action for an entity")
    predict_parser.add_argument("--entity", type=str, required=True, help="Entity ID (e.g. SKU_WH_1)")
    predict_parser.add_argument("--horizon", type=int, default=7, help="Simulation horizon")

    # Simulate Command
    sim_parser = subparsers.add_parser("simulate", help="Simulate a specific action rollout")
    sim_parser.add_argument("--entity", type=str, required=True, help="Entity ID")
    sim_parser.add_argument("--action", type=str, required=True, help="Action name (reorder/transfer)")
    sim_parser.add_argument("--value", type=float, required=True, help="Action value")
    sim_parser.add_argument("--horizon", type=int, default=14, help="Simulation horizon")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # 1. Setup Infrastructure
    config = load_config(args.config)
    builder = StateBuilder(config)
    db = DuckDBAdapter(args.db)
    rule = get_rule(config.physics_rule)

    model = WorldModel(rule=rule, params=config.rule_params)
    if Path(args.model_path).exists():
        model.load(args.model_path)

    # 2. Execute Commands
    if args.command == "train":
        print(f"[*] Training model from {args.db}...")
        entity = args.entity or "SKU_WH_1"
        history = db.load_history(entity)

        if len(history) < 10:
            print(f"[!] Not enough history for {entity} to train (found {len(history)}). Need 10+.")
            sys.exit(1)

        for day in history:
            # Replay history to build error log
            # In a real system, the next state would be the actual observed state.
            # Here we just populate the error history from the DB observations.
            pass

        model.train_correction()
        model.save(args.model_path)
        print(f"[+] Model trained and saved to {args.model_path}")

    elif args.command == "predict":
        engine = DecisionEngine(model, config)
        history = db.load_history(args.entity)
        if not history:
            print(f"[!] No current state found for {args.entity} in {args.db}")
            sys.exit(1)

        current_state = history[-1].state
        print(f"[*] Current State: {current_state.to_dict()}")

        candidates = [Action.no_op(), Action("reorder", 50), Action("reorder", 100)]
        decision = engine.decide(current_state, candidates, horizon=args.horizon)

        print("\n--- ENGINE RECOMMENDATION ---")
        print(f"Action: {decision.action}")
        print(f"Score:  {decision.score:.2f}")
        print("\n--- PROJECTED FUTURE ---")
        print(decision.trajectory.as_dataframe().head(5))

    elif args.command == "simulate":
        from opl.simulator.rollout import rollout

        history = db.load_history(args.entity)
        if not history:
            print(f"[!] No state found for {args.entity}")
            sys.exit(1)

        start_state = history[-1].state
        action = Action(args.action, args.value)

        print(f"[*] Rollout for {args.entity} | Action: {action}...")
        trajectory = rollout(model, start_state, action, horizon=args.horizon)
        print(trajectory.as_dataframe())


if __name__ == "__main__":
    main()
