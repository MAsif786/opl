"""
Configuration Schema — Defines how to port the engine to a new industry.

Using Pydantic, we define a strict schema for domain configuration files (YAML).
This allows users to define the entities, dimensions, actions, and cost parameters
without writing Python code, making the core engine entirely reusable across industries.
"""

from pydantic import BaseModel, Field


class DomainConfig(BaseModel):
    """Configuration for a specific industry/domain application of the engine.

    This replaces hardcoded dimensions and costs.
    """

    domain: str = Field(..., description="Name of the domain, e.g., 'logistics', 'finance'")

    dimensions: list[str] = Field(
        ..., min_length=1, description="Ordered list of state dimension names (e.g., ['stock', 'demand'])"
    )

    categorical_dimensions: list[str] = Field(
        default_factory=list, description="List of dimensions that are categorical"
    )

    physics_rule: str = Field("logistics_basic", description="Name of the physics rule to use from the Rule Registry")

    rule_params: dict[str, float] = Field(
        default_factory=dict, description="Physical constants for the physics rule (e.g., lead_time)"
    )

    actions: list[str] = Field(
        ..., min_length=1, description="List of allowed action names (e.g., ['reorder', 'transfer'])"
    )

    cost_params: dict[str, float] = Field(
        default_factory=dict, description="Key-value pairs for the domain's cost function"
    )
