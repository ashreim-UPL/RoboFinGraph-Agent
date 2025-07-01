# utils/config.utils.py

from typing import Dict, Any, List
import os
from utils.logger import get_logger, log_event

logger = get_logger()


def resolve_model_config(model_name: str, model_providers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Find the provider entry for `model_name` and return its config.
    Raises ValueError (and logs) if not found.
    """
    for provider in model_providers:
        if "models" in provider and model_name in provider["models"]:
            config = {
                "model": model_name,
                "api_key": provider.get("api_key", ""),
                "api_type": provider.get("provider", "")
            }
            if "base_url" in provider:
                config["base_url"] = provider["base_url"]

            logger.info(f"Resolved config for model '{model_name}' from provider '{provider.get('provider')}'")
            log_event("model_config_resolved", config)
            return config

    err_msg = f"Model '{model_name}' not found in model_providers."
    logger.error(err_msg)
    log_event("model_config_error", {"model": model_name, "error": err_msg})
    raise ValueError(err_msg)


def inject_model_env(model_config: Dict[str, Any]):
    """
    Injects the API key (and base_url if needed) into environment vars
    and logs the injection event.
    """
    api_type = model_config.get("api_type", "").lower()
    model_name = model_config.get("model", "<unknown>")

    if api_type == "openai":
        os.environ["OPENAI_API_KEY"] = model_config.get("api_key", "")
        msg = f"Injected OPENAI_API_KEY for model: {model_name}"
        logger.info(msg)
        log_event("env_injection", {"model": model_name, "api": "openai"})

    elif api_type == "together":
        os.environ["TOGETHER_API_KEY"] = model_config.get("api_key", "")
        os.environ["TOGETHER_BASE_URL"] = model_config.get("base_url", "")
        msg = f"Injected TOGETHER_API_KEY and BASE_URL for model: {model_name}"
        logger.info(msg)
        log_event("env_injection", {
            "model": model_name,
            "api": "together",
            "base_url": model_config.get("base_url", "")
        })

    else:
        msg = f"No env rules for API type: '{api_type}' (model: {model_name})"
        logger.info(msg)
        log_event("env_injection_skipped", {"model": model_name, "api_type": api_type})
