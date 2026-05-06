from __future__ import annotations

from typing import TYPE_CHECKING
from opl.model.action import Action
from opl.state.vector import StateVector

if TYPE_CHECKING:
    from opl.model.world_model import RuleFunction

# ─── Indices ──────────────────────────────────────────────────────────────────

# Core dimensions (Standard)
_STOCK = 0
_DEMAND = 1
_INCOMING = 2
_DELAY = 3

# Extended dimensions (Multi-WH / Dynamic)
_OTHER_WH_STOCK = 4
_TRANSPORT_TIME = 5


# ─── Rules ────────────────────────────────────────────────────────────────────

def simple_stock_rule(s_t: StateVector, action: Action, params: dict | None = None) -> StateVector:
    """
    Base physics rule for warehouse stock evolution.
    
    Logic:
    - Stock decreases by demand.
    - If delay hits 0, incoming arrives.
    - Reorder action sets delay based on transport_time (if present) or params.
    """
    params = params or {}
    
    # Dynamic lead time: Look in state first, then params, then default
    if len(s_t) > _TRANSPORT_TIME:
        lead_time = s_t[_TRANSPORT_TIME]
    else:
        lead_time = params.get("lead_time", 3.0)

    stock = s_t[_STOCK]
    demand = s_t[_DEMAND]
    incoming = s_t[_INCOMING]
    delay = s_t[_DELAY]

    # Arrival logic
    if delay <= 0 and incoming > 0:
        stock = stock - demand + incoming
        incoming = 0.0
        delay = 0.0
    else:
        stock = stock - demand
        delay = max(0.0, delay - 1)

    # Reorder logic
    if action.name == "reorder" and action.value > 0:
        incoming = action.value
        delay = lead_time

    # Construct next state (preserve all dimensions)
    next_values = list(s_t.values)
    next_values[_STOCK] = stock
    next_values[_INCOMING] = incoming
    next_values[_DELAY] = delay
    # Demand and others persist by default

    return StateVector(next_values, names=s_t.names, metadata=s_t.metadata)


def multi_warehouse_rule(s_t: StateVector, action: Action, params: dict | None = None) -> StateVector:
    """
    Physics rule for warehouse transfers and reorders.
    
    Logic:
    - Normal stock/demand/arrival logic.
    - Transfer action moves stock from other_wh_stock to transit.
    """
    # Reuse base logic for the first 4 dims
    next_state = simple_stock_rule(s_t, action, params)
    
    next_values = list(next_state.values)
    other_wh_stock = s_t[_OTHER_WH_STOCK]
    transport_time = s_t[_TRANSPORT_TIME]

    # Specific Transfer Logic
    if action.name == "transfer" and action.value > 0:
        actual_transfer = min(action.value, other_wh_stock)
        other_wh_stock -= actual_transfer
        next_values[_INCOMING] = actual_transfer
        next_values[_DELAY] = transport_time

    next_values[_OTHER_WH_STOCK] = other_wh_stock
    # transport_time persists

    return StateVector(next_values, names=s_t.names, metadata=s_t.metadata)


# ─── Registry ─────────────────────────────────────────────────────────────────

RULE_REGISTRY = {
    "logistics_basic": simple_stock_rule,
    "logistics_multi_warehouse": multi_warehouse_rule,
}

def get_rule(name: str) -> RuleFunction:
    if name not in RULE_REGISTRY:
        raise ValueError(f"Rule '{name}' not found. Available: {list(RULE_REGISTRY.keys())}")
    return RULE_REGISTRY[name]
