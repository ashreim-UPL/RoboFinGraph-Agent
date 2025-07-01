# main.py

import argparse
import json
import os
import sys
import io
import traceback
from typing import Dict, Any, List

from utils.logger import get_logger, log_event
from utils.config_utils import resolve_model_config, inject_model_env

logger = get_logger()

# Ensure cache directory exists
if not os.path.isdir(".cache"):
    os.makedirs(".cache")


def load_config_into_environ(config_path: str) -> Dict[str, Any]:
    """
    Loads the full configuration from a JSON file.
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            full_config = json.load(f)
        if not isinstance(full_config, dict):
            msg = f"Invalid configuration format in {config_path}: expected a dict."
            logger.error(msg)
            log_event("config_load_error", {"path": config_path, "error": msg})
            return {}
        logger.info(f"Config loaded from {config_path}")
        log_event("config_loaded", {"path": config_path})
        return full_config

    except FileNotFoundError:
        msg = f"Config file not found at {config_path}"
        logger.error(msg)
        log_event("config_load_error", {"path": config_path, "error": msg})
        return {}

    except json.JSONDecodeError as e:
        msg = f"JSON decode error in {config_path}: {e}"
        logger.error(msg)
        log_event("config_load_error", {"path": config_path, "error": msg})
        return {}

    except Exception as e:
        msg = f"Unexpected error loading config: {e}"
        logger.error(msg)
        log_event("config_load_error", {"path": config_path, "error": msg})
        return {}


def main():
    parser = argparse.ArgumentParser(description="FinRobot Annual Report Orchestration")
    parser.add_argument("company", type=str, help="Company name (e.g., 'Apple Inc.')")
    parser.add_argument("year", type=str, help="Year to analyze (e.g., '2023')")
    parser.add_argument(
        "--config",
        type=str,
        default="finrobot_config.json",
        help="Path to the config JSON file."
    )
    parser.add_argument(
        "--llm_models",
        type=str,
        default="{}",
        help="JSON string of agent→model mappings."
    )
    parser.add_argument(
        "--report_type",
        type=str,
        default="kpi_bullet_insights",
        help="Type of report to generate"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    logger.info(f"Starting orchestration for {args.company} ({args.year})")
    log_event("orchestration_start", {
        "company": args.company,
        "year": args.year,
        "report_type": args.report_type,
        "verbose": args.verbose
    })

    # 1. Load config
    app_config = load_config_into_environ(args.config)
    if not app_config:
        logger.error("Aborting: could not load configuration.")
        sys.exit(1)

    # 2. Inject raw API keys
    for key_name, key_value in app_config.get("api_keys", {}).items():
        if key_name and key_value:
            os.environ[key_name] = key_value
            logger.info(f"Injected API key: {key_name}")
            log_event("api_key_injected", {"key": key_name})
        else:
            logger.warning(f"Skipping empty API key entry: {key_name}")
            log_event("api_key_skipped", {"key": key_name})
    print(f"\n Launching FinRobot orchestration for: {args.company} ({args.year})\n")
    # 3. Parse frontend LLM models override
    try:
        frontend_llm = json.loads(args.llm_models)
        logger.info(f"Frontend LLM models: {frontend_llm}")
        log_event("frontend_llm_parsed", {"models": frontend_llm})
    except json.JSONDecodeError as e:
        msg = f"Error parsing --llm_models JSON: {e}"
        logger.error(msg)
        log_event("frontend_llm_error", {"error": msg})
        frontend_llm = {}

    # 4. Merge LLM models
    final_llm = app_config.get("llm_models", {}).copy()
    final_llm.update(frontend_llm)
    app_config["llm_models"] = final_llm
    logger.info(f"Final LLM models config: {final_llm}")
    log_event("llm_models_configured", {"models": final_llm})

    # 5. Preload env for each LLM model
    for agent_name, model_name in final_llm.items():
        if not model_name:
            continue
        try:
            model_cfg = resolve_model_config(model_name, app_config.get("model_providers", []))
            inject_model_env(model_cfg)
            logger.info(f"Env preloaded for agent '{agent_name}' → model '{model_name}'")
            log_event("model_env_preloaded", {"agent": agent_name, "model": model_name})
        except Exception as e:
            msg = f"Failed to preload env for '{agent_name}': {e}"
            logger.error(msg)
            log_event("model_env_error", {"agent": agent_name, "error": str(e)})

    # 6. Run orchestration
    try:
        from workflows.orchestrator import run_orchestration
        run_orchestration(
            company=args.company,
            year=args.year,
            config=app_config,
            report_type=args.report_type,
            verbose=args.verbose
        )
        print("\n✅ Orchestration completed. Check logs and output files.\n")
        logger.info("Orchestration completed successfully.")
        log_event("orchestration_completed", {"company": args.company, "year": args.year})

    except ImportError as e:
        msg = "Module 'workflows.orchestrator' not found."
        logger.error(msg)
        log_event("orchestration_import_error", {"error": str(e)})
        sys.exit(1)

    except Exception as e:
        err_trace = traceback.format_exc()
        logger.error(f"Orchestration failed: {e}")
        log_event("orchestration_failed", {"error": str(e), "traceback": err_trace})
        sys.exit(1)


if __name__ == "__main__":
    # Ensure stdout is UTF-8
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    main()
