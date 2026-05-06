# Cognitive Decision Engine (OPL)

The **Cognitive Decision Engine** is an industry-agnostic, hybrid AI framework that recommends optimal operational decisions by forecasting future consequences.

Instead of relying purely on opaque "black-box" machine learning or rigid deterministic rules, this engine combines both. It uses a **physics engine** to handle known domain rules (e.g., inventory math) and an **ML correction layer** to learn systematic biases (e.g., spoilage, unpredictable delays) from historical data. It then simulates thousands of possible futures and selects the decision that minimizes long-term costs.

---

## 🧠 Core Architecture

The architecture is fully decoupled, mathematically strict, and entirely test-driven.

1. **State Vector (`src/opl/state/`)**: 
   The atomic unit of the system. Represents entities as immutable 1D/2D NumPy arrays.
2. **World Model (`src/opl/model/`)**: 
   A hybrid system: `Future = Physics_Rule + ML_Correction`.
   - **Rule Registry**: Physics logic (e.g., `logistics_basic`) is selectable via YAML and extensible.
   - **Model Persistence**: Trained ML weights are serialized to `.joblib` files, allowing for instant "warm starts" without retraining on history every time.
3. **Simulator (`src/opl/simulator/`)**:
   Rolls the `WorldModel` forward across a time horizon.
   - **Batched Execution**: Supports 100,000+ entities concurrently using 2D matrix math.
4. **Data Adapters (`src/opl/adapter/`)**:
   - **CSV Adapter**: For rapid prototyping and local research.
   - **DuckDB SQL Adapter**: Production-grade persistence. Logs every real-world observation into a local SQL database for auditing and automated retraining loops.
5. **Cost Evaluator (`src/opl/evaluator/`)**:
   Assigns financial scores using non-linear step functions (Warehouse Capacity, Bulk Discounts, Safety Stock Buffers).

---

## ⚙️ How It Works (The Core Loop)

1. **Initialization**: The engine loads the `domain.yaml` and connects to the **DuckDB** database.
2. **Warm Start**: It checks for a pre-trained `.joblib` model. If found, it skips the cold start training.
3. **Rollout Generation**: Evaluates every possible action using the `BatchedSimulator`.
4. **Cost Assignment**: Applies the domain's Cost Function, penalizing stockouts, capacity overflows, and safety stock violations.
5. **Audit Logging**: Saves today's decision and the eventual outcome back to **DuckDB** to improve tomorrow's accuracy.

---

## 🛠️ Configuration & Portability

The engine is industry-agnostic. To port it, define a new YAML:

**Example: `data/domain_logistics.yaml`**
```yaml
domain_name: logistics_replenishment
physics_rule: logistics_basic
rule_params:
  lead_time: 4.0
action_space:
  type: discrete
  options: [no_op, reorder]
cost_params:
  stockout_penalty: 1500.0
  safety_stock_threshold: 50.0
  warehouse_capacity: 500.0
```

---

## 🚀 Running the Engine

```bash
# Train and Save a new model
python src/opl/cli.py --data history.csv --config domain.yaml --model my_model.joblib

# Load an existing model (Instant execution)
python src/opl/cli.py --data today.csv --config domain.yaml --model my_model.joblib
```


**Output Example:**
```text
🎯 RECOMMENDED DECISION
==================================================
Action to take: REORDER 100
Expected Cost:  1254.50
--------------------------------------------------
Future Trajectory (Next 5 Days):
  Day 1: Stock = 40.0 | Expected Demand = 20.0
  Day 2: Stock = 20.0 | Expected Demand = 20.0
  Day 3: Stock = 100.0 | Expected Demand = 20.0  <- Order arrives
  Day 4: Stock = 80.0 | Expected Demand = 20.0
```

---

## 🧪 Testing & Scale

The system is built using strict Test-Driven Development (TDD).
To run the full suite:
```bash
make test
```

The Non-Functional Requirement (NFR) of the system requires it to scale to **100,000 entities** in under 1 minute. You can verify this math via the benchmark test:
```bash
pytest tests/unit/test_scale.py -v -s
```
*(Spoilers: It simulates 100k entities for 14 days in ~0.02 seconds).*
