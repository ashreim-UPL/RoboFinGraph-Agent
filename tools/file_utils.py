# tools/file_utils.py

import os
import json
from typing import Dict, Any
from PIL import Image

def read_text_file(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def get_file_size(filepath: str) -> int:
    return os.path.getsize(filepath)

def list_directory_files(directory: str):
    return os.listdir(directory)

def get_file_metadata(filepath: str):
    return {
        "size": os.path.getsize(filepath),
        "last_modified": os.path.getmtime(filepath),
        "is_file": os.path.isfile(filepath),
        "extension": os.path.splitext(filepath)[1]
    }

def read_image_metadata(filepath: str):
    with Image.open(filepath) as img:
        return {
            "format": img.format,
            "width": img.width,
            "height": img.height,
            "mode": img.mode
        }

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

def check_file_stats(filepath: str) -> Dict[str, str]:
    """
    Return basic stats about the file: size (KB), type, and for JSON/TXT, a content preview.
    """
    try:
        size_kb = os.path.getsize(filepath) / 1024
        _, ext = os.path.splitext(filepath)
        ext = ext.lower()

        info = {
            "size_kb": f"{size_kb:.2f}",
            "type": ext.replace('.', '')
        }

        if ext == ".json":
            with open(filepath, "r", encoding="utf-8") as f:
                content = json.load(f)
            info["content_keys"] = list(content.keys()) if isinstance(content, dict) else f"len={len(content)}"

        elif ext == ".txt":
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
            preview = "".join(lines[:5]).strip()
            info["preview"] = preview[:300] + ("..." if len(preview) > 300 else "")

        elif ext == ".png":
            info["png_bytes"] = f"{size_kb:.2f} KB"
            # If you want to verify integrity later, consider checking dimensions using PIL

        return info

    except Exception as e:
        return {"error": str(e)}
