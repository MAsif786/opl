"""
Rules — Domain physics encoded as pure functions.

A rule is a function: (StateVector, Action) → StateVector
It encodes deterministic domain knowledge ("stock goes down by demand").

The ML correction layer will learn the systematic errors in these rules,
so they don't need to be perfect — just structurally correct.

Design decisions:
- Pure functions, no side effects, no state
- Named dimensions accessed by index for speed (names for debug)
- Default lead time is a constant — adapter can override later
"""

from __future__ import annotations

from opl.model.action import Action
from opl.state.vector import StateVector

# State dimension indices for logistics
_STOCK = 0
_DEMAND = 1
_INCOMING = 2
_DELAY = 3

# Default lead time for reorders (days)
DEFAULT_LEAD_TIME = 3


def simple_stock_rule(s_t: StateVector, action: Action, params: dict | None = None) -> StateVector:
    """Base physics rule for warehouse stock evolution.
    
    Params:
        lead_time: The delay in days for a reorder to arrive (default 3).
    """
    params = params or {}
    lead_time = params.get("lead_time", 3)

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

    # Demand persists
    next_demand = demand

    values = [stock, next_demand, incoming, delay]
    return StateVector(values, names=s_t.names)



# State dimension indices for multi-warehouse
_M_STOCK = 0
_M_DEMAND = 1
_M_INCOMING = 2
_M_DELAY = 3
_M_OTHER_WH_STOCK = 4
_M_TRANSPORT_TIME = 5

def multi_warehouse_rule(s_t: StateVector, action: Action) -> StateVector:
    """Multi-entity physics rule for warehouse transfers.

    State dimensions: [local_stock, demand, incoming, delay, other_wh_stock, transport_time]

    Logic:
        - local_stock decreases by demand
        - if delay == 0: local_stock increases by incoming, incoming resets
        - delay decrements by 1 (min 0)
        - if transfer action: 
            take from other_wh_stock (capped at available)
            set up incoming and delay

    Args:
        s_t: Current state vector.
        action: Action to apply.

    Returns:
        Predicted next state vector.
    """
    local_stock = s_t[_M_STOCK]
    demand = s_t[_M_DEMAND]
    incoming = s_t[_M_INCOMING]
    delay = s_t[_M_DELAY]
    other_wh_stock = s_t[_M_OTHER_WH_STOCK]
    transport_time = s_t[_M_TRANSPORT_TIME]

    # Arrival logic for local warehouse
    if delay <= 0 and incoming > 0:
        local_stock = local_stock - demand + incoming
        incoming = 0.0
        delay = 0.0
    else:
        local_stock = local_stock - demand
        delay = max(0.0, delay - 1)

    # Transfer logic: if action is transfer, move stock from B to transit
    if action.name == "transfer" and action.value > 0:
        actual_transfer = min(action.value, other_wh_stock)
        other_wh_stock -= actual_transfer
        incoming = actual_transfer
        delay = transport_time

    # Demand and transport time persist
    next_demand = demand
    next_transport_time = transport_time

    values = [local_stock, next_demand, incoming, delay, other_wh_stock, next_transport_time]
    return StateVector(values, names=s_t.names)


# Rule Registry — Allows dynamic lookup from configuration
RULE_REGISTRY = {
    "logistics_basic": simple_stock_rule,
    "logistics_multi_warehouse": multi_warehouse_rule,
}


def get_rule(name: str) -> RuleFunction:
    """Get a physics rule by name.

    Args:
        name: Name of the rule in the registry.

    Returns:
        The rule function.

    Raises:
        ValueError: If rule name is not found.
    """
    if name not in RULE_REGISTRY:
        raise ValueError(
            f"Rule '{name}' not found in registry. "
            f"Available: {list(RULE_REGISTRY.keys())}"
        )
    return RULE_REGISTRY[name]

