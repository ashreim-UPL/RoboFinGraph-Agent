# tools/file_utils.py

import os
import json
from typing import Dict, Any

def read_json_file(file_path: str) -> Any:
    """
    Reads a JSON file from disk and returns its content.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_to_file(data: Any, file_path: str = "output.json") -> str:
    """
    Saves a Python object to disk as JSON.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return file_path

def load_config_into_environ(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        config = json.load(f)

    # Recursively find and replace env placeholders
    def resolve_env_vars(cfg):
        if isinstance(cfg, dict):
            for key, value in cfg.items():
                cfg[key] = resolve_env_vars(value)
        elif isinstance(cfg, list):
            for i, item in enumerate(cfg):
                cfg[i] = resolve_env_vars(item)
        elif isinstance(cfg, str) and cfg.startswith("env:"):
            env_var_name = cfg.split(":", 1)[1]
            return os.getenv(env_var_name)
        return cfg

    config = resolve_env_vars(config)
    
    # Set other API keys as environment variables for tool access
    if "api_keys" in config:
        for key, value in config["api_keys"].items():
            os.environ[key] = value

    return config
