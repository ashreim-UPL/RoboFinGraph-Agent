# main.py

import argparse
import json
import os
import sys
import io
import traceback
from typing import Dict, Any, List
from tabulate import tabulate

from utils.logger import get_logger, log_event
from utils.config_utils import resolve_model_config, inject_model_env

logger = get_logger()

# Ensure cache directory exists
if not os.path.isdir(".cache"):
    os.makedirs(".cache")


def load_config_into_environ(config_path: str) -> Dict[str, Any]:
    """
    Loads the full configuration from a JSON file,
    and injects all API keys from both 'api_keys' and 'providers'
    into the environment.
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

        # Inject top-level API keys
        for k, v in full_config.get("api_keys", {}).items():
            if k and v:
                os.environ[k] = v
                logger.info(f"Injected API key: {k}")
                log_event("api_key_injected", {"key": k})

        # Inject provider API keys (standardized env var names)
        for provider_name, provider_data in full_config.get("providers", {}).items():
            env_var = f"{provider_name.upper()}_API_KEY"
            api_key = provider_data.get("api_key")
            if api_key:
                os.environ[env_var] = api_key
                logger.info(f"Injected Provider API key: {env_var}")
                log_event("provider_api_key_injected", {"env_var": env_var})

            # Optionally: inject provider-specific base URLs as well
            if "base_url" in provider_data:
                url_env = f"{provider_name.upper()}_BASE_URL"
                os.environ[url_env] = provider_data["base_url"]
                logger.info(f"Injected {url_env}")
                log_event("provider_base_url_injected", {"env_var": url_env})

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
        default="langgraph_config.json",
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
        
        final_state= run_orchestration(
            company=args.company,
            year=args.year,
            config=app_config,
            report_type=args.report_type,
            verbose=args.verbose
        )

        eval = final_state.memory["final_evaluation"]

        # Print ICAIF scores
        print("\n=== ICAIF Ratings ===")
        print(tabulate(eval["icaif_scores"].items(),
                    headers=["Criterion", "Score"],
                    tablefmt="github"))

        # Print pipeline matrix
        print("\n=== Pipeline Matrix ===")
        print(tabulate(eval["pipeline_matrix"],
                    headers=["Agent", "Requests", "Avg Latency (ms)", "Errors"],
                    tablefmt="github"))
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
