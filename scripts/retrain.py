"""
OPL Retraining Job — Updates the ML model using history stored in DuckDB.

Usage:
    python scripts/retrain.py --db data/opl.db --config data/domain_logistics.yaml --output model.joblib
"""

import argparse
import yaml
from opl.adapter.duckdb_adapter import DuckDBAdapter
from opl.config.schema import DomainConfig
from opl.model.world_model import WorldModel
from opl.model.rules import get_rule

def main():
    parser = argparse.ArgumentParser(description="Retrain OPL Model from DuckDB History")
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    parser.add_argument("--config", required=True, help="Path to domain YAML")
    parser.add_argument("--output", required=True, help="Where to save the updated .joblib model")
    args = parser.parse_args()

    # 1. Load Config
    with open(args.config, "r") as f:
        config_data = yaml.safe_load(f)
        config = DomainConfig(**config_data)

    print(f"🔄 Starting Retraining Job for Domain: {config.domain}")

    # 2. Connect to DB and Load History
    db = DuckDBAdapter(args.db)
    history = db.load_history(config.domain)
    db.close()

    if len(history) < 10:
        print(f"⚠️  Not enough data in DB to retrain. Found {len(history)} records, need at least 10.")
        return

    print(f"📊 Loaded {len(history)} observations from DuckDB.")

    # 3. Initialize Model
    physics_rule = get_rule(config.physics_rule)
    world_model = WorldModel(rule=physics_rule, rule_params=config.rule_params)

    # 4. Retrain
    print("🧠 Training ML correction layer...")
    world_model.train_correction(history)

    # 5. Save
    print(f"💾 Saving updated model to {args.output}...")
    world_model.save(args.output)
    
    print("✅ Retraining complete. The live engine will pick up the new model on next boot.")

if __name__ == "__main__":
    main()
