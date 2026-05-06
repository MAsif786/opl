"""
TDD Tests for Phase 3: YAML Configuration & Domain Portability.

These tests prove FR3/FR4: We can define a completely new domain (e.g., Finance
or Healthcare) using only a YAML file, without rewriting the core engine.

Tests written BEFORE implementation.
"""

import pytest
import yaml

from opl.config.schema import DomainConfig
from opl.state.builder import StateBuilder


@pytest.fixture
def finance_yaml():
    """A completely different industry schema defined in YAML."""
    return """
    domain: finance_treasury
    dimensions:
      - account_balance
      - daily_burn
      - pending_inbound
      - days_to_clear
    actions:
      - transfer_in
      - hold
    cost_params:
      overdraft_penalty: 5000.0
      idle_cash_cost: 0.05
    """


class TestConfig:
    """Testing the parsing and usage of domain configs."""

    @pytest.mark.unit
    def test_parses_yaml_into_pydantic_schema(self, finance_yaml):
        """Pydantic should correctly load the YAML into a strong type."""
        raw_dict = yaml.safe_load(finance_yaml)
        config = DomainConfig(**raw_dict)

        assert config.domain == "finance_treasury"
        assert len(config.dimensions) == 4
        assert "account_balance" in config.dimensions
        assert "transfer_in" in config.actions
        assert config.cost_params["overdraft_penalty"] == 5000.0

    @pytest.mark.unit
    def test_creates_state_builder_from_config(self, finance_yaml):
        """We should be able to instantiate a StateBuilder directly from the config."""
        raw_dict = yaml.safe_load(finance_yaml)
        config = DomainConfig(**raw_dict)

        # StateBuilder is configured with the dimensions from YAML
        builder = StateBuilder(fields=config.dimensions)

        raw_data = {
            "account_balance": 100000,
            "daily_burn": 5000,
            "pending_inbound": 0,
            "days_to_clear": 0
        }

        state = builder.build(raw_data)
        assert state.names == ("account_balance", "daily_burn", "pending_inbound", "days_to_clear")
        assert state[0] == 100000.0

    @pytest.mark.unit
    def test_rejects_invalid_config(self):
        """Pydantic must fail if required fields like dimensions are missing."""
        bad_yaml = """
        domain: bad_domain
        actions: [do_thing]
        """
        raw_dict = yaml.safe_load(bad_yaml)
        with pytest.raises(ValueError):
            DomainConfig(**raw_dict)
