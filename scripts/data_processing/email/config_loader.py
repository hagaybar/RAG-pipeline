import os
import yaml
from typing import Any

class ConfigLoader:
    """A class to load and access configuration from a YAML file."""

    def __init__(self, config_file: str = "config.yaml") -> None:
        """
        Initialize the ConfigLoader with the given file path.

        Args:
            config_file (str): Path to the YAML configuration file.
        """
        self.config_file = config_file
        self.config_path = config_file
        self.config_data = self._load_config()

    def _load_config(self) -> dict:
        """Load configuration from the YAML file with validation."""
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Config file not found: {self.config_file}")

        with open(self.config_file, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
            if not data:
                raise ValueError(f"Configuration file '{self.config_file}' is empty or malformed.")
            return data



    def get(self, key_path: str) -> Any:
        """
        Retrieve a nested configuration value using dot notation (e.g., 'paths.data_dir').

        Args:
            key_path (str): Dot-separated path to the config key.

        Returns:
            Any: The value associated with the given key path.

        Raises:
            KeyError: If any part of the key path is missing.
        """
        keys = key_path.split(".")
        value = self.config_data
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                raise KeyError(f"Missing configuration key: '{key}' in path '{key_path}'")
            value = value[key]
        return value
