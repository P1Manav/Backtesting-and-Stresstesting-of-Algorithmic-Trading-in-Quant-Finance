"""Configuration loading utilities"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
class ConfigLoader:
    """Load and manage stress testing configuration files."""
    def __init__(self, config_dir: Optional[str] = None):
"""Initialize config loader. Args: config_dir: Directory containing config files. Defaults to ./config/"""
        if config_dir is None:
            config_dir = Path(__file__).parent.parent / "config"
        self.config_dir = Path(config_dir)
        self.stress_config = None
        self.scenario_params = None
    def load_stress_config(self) -> Dict[str, Any]:
        """Load main stress testing configuration."""
        config_path = self.config_dir / "stress_config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        with open(config_path, 'r') as f:
            self.stress_config = yaml.safe_load(f)
        return self.stress_config
    def load_scenario_parameters(self) -> Dict[str, Any]:
        """Load scenario-specific parameters."""
        params_path = self.config_dir / "scenario_parameters.yaml"
        if not params_path.exists():
            raise FileNotFoundError(f"Parameters not found: {params_path}")
        with open(params_path, 'r') as f:
            self.scenario_params = yaml.safe_load(f)
        return self.scenario_params
    def load_all(self) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Load both configuration files."""
        stress = self.load_stress_config()
        params = self.load_scenario_parameters()
        return stress, params
    def get_scenario_config(self, scenario_name: str) -> Dict[str, Any]:
        """Get configuration for a specific scenario."""
        if not self.stress_config:
            self.load_stress_config()
        scenarios = self.stress_config.get('stress_testing', {}).get('scenarios', {})
        return scenarios.get(scenario_name, {})
    def get_scenario_enabled(self, scenario_name: str) -> bool:
        """Check if a scenario is enabled."""
        config = self.get_scenario_config(scenario_name)
        return config.get('enabled', False)
    def get_baseline_config(self) -> Dict[str, Any]:
        """Get baseline testing configuration."""
        if not self.stress_config:
            self.load_stress_config()
        return self.stress_config.get('stress_testing', {}).get('baseline', {})

