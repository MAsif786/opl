import yaml
from pathlib import Path
from .schema import DomainConfig

def load_config(path: str | Path) -> DomainConfig:
    """Load domain configuration from a YAML file."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return DomainConfig(**data)
