"""Configuration loader for the Confluence RAG system."""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import structlog
import yaml
from pydantic import ValidationError

from src.models.config import AppConfig

log = structlog.stdlib.get_logger()


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing."""

    pass


class ConfigLoader:
    """Loads and validates application configuration from YAML files and environment variables."""

    def __init__(self) -> None:
        """Initialize the ConfigLoader."""
        self.env_var_pattern = re.compile(r"\$\{([^}]+)\}")

    def load_config(self, config_path: Optional[str] = None) -> AppConfig:
        """Load configuration from YAML file with environment variable overrides.

        Args:
            config_path: Path to the configuration YAML file. If None, uses default.yaml

        Returns:
            AppConfig: Validated application configuration

        Raises:
            ConfigurationError: If configuration file is missing or invalid
            ValidationError: If configuration validation fails
        """
        # Determine config file path
        if config_path is None:
            config_path = self._get_default_config_path()

        log.info("loading_configuration", config_path=config_path)

        # Load YAML file
        config_dict = self._load_yaml_file(config_path)

        # Substitute environment variables
        config_dict = self._substitute_env_vars(config_dict)

        # Validate and create AppConfig
        try:
            app_config = AppConfig(**config_dict)
            log.info("configuration_loaded_successfully")
            return app_config
        except ValidationError as e:
            log.error("configuration_validation_failed", error=str(e))
            raise ConfigurationError(f"Configuration validation failed: {e}") from e

    def _get_default_config_path(self) -> str:
        """Get the default configuration file path based on environment.

        Returns:
            str: Path to the configuration file
        """
        env = os.getenv("APP_ENV", "default")
        config_dir = Path(__file__).parent.parent.parent / "config"
        config_file = config_dir / f"{env}.yaml"

        if not config_file.exists():
            # Fall back to default.yaml
            config_file = config_dir / "default.yaml"

        if not config_file.exists():
            raise ConfigurationError(
                f"Configuration file not found: {config_file}. "
                f"Please create config/default.yaml or set APP_ENV to a valid environment."
            )

        return str(config_file)

    def _load_yaml_file(self, config_path: str) -> Dict[str, Any]:
        """Load YAML configuration file.

        Args:
            config_path: Path to the YAML file

        Returns:
            Dict containing the configuration

        Raises:
            ConfigurationError: If file cannot be read or parsed
        """
        try:
            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f)

            if config_dict is None:
                raise ConfigurationError(f"Configuration file is empty: {config_path}")

            log.debug("yaml_file_loaded", config_path=config_path)
            return config_dict
        except FileNotFoundError:
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Failed to parse YAML file {config_path}: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration file {config_path}: {e}")

    def _substitute_env_vars(self, config: Any) -> Any:
        """Recursively substitute environment variables in configuration.

        Environment variables are specified as ${VAR_NAME} in the YAML file.

        Args:
            config: Configuration value (can be dict, list, str, or other types)

        Returns:
            Configuration with environment variables substituted

        Raises:
            ConfigurationError: If a required environment variable is not set
        """
        if isinstance(config, dict):
            return {key: self._substitute_env_vars(value) for key, value in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str):
            return self._substitute_env_var_in_string(config)
        else:
            return config

    def _substitute_env_var_in_string(self, value: str) -> str:
        """Substitute environment variables in a string.

        Args:
            value: String that may contain ${VAR_NAME} patterns

        Returns:
            String with environment variables substituted

        Raises:
            ConfigurationError: If a required environment variable is not set
        """
        matches = self.env_var_pattern.findall(value)

        for var_name in matches:
            env_value = os.getenv(var_name)
            if env_value is None:
                raise ConfigurationError(
                    f"Required environment variable not set: {var_name}. "
                    f"Please set {var_name} in your environment or .env file."
                )
            value = value.replace(f"${{{var_name}}}", env_value)

        return value

    def validate_config(self, config: AppConfig) -> list[str]:
        """Validate configuration and return any warnings.

        Note: Pydantic handles most validation automatically during model creation.
        This method can be used for additional custom validation logic.

        Args:
            config: Application configuration to validate

        Returns:
            List of warning messages (empty if no warnings)
        """
        warnings = []

        # Check chunk overlap is reasonable relative to chunk size
        if config.processing.chunk_overlap >= config.processing.chunk_size:
            warnings.append(
                f"chunk_overlap ({config.processing.chunk_overlap}) should be less than "
                f"chunk_size ({config.processing.chunk_size})"
            )

        # Check vector store type is supported
        supported_stores = ["chroma", "faiss", "qdrant"]
        if config.vector_store.type not in supported_stores:
            warnings.append(
                f"vector_store.type '{config.vector_store.type}' may not be supported. "
                f"Supported types: {supported_stores}"
            )

        if warnings:
            log.warning("configuration_validation_warnings", warnings=warnings)

        return warnings
