from pathlib import Path
from typing import Any, Dict
import yaml

def load_yaml(config_path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file safely."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_all_configs(configs_dir: str | Path = "configs") -> Dict[str, Any]:
    """Load all YAML configuration files in the config directory."""
    cdir = Path(configs_dir)
    configs = {}
    for yml in cdir.glob("*.yaml"):
        configs[yml.stem] = load_yaml(yml)
    return configs
