# main.py

import argparse
import json
import os
import sys
import io
import traceback
from typing import Dict, Any
from tabulate import tabulate

from utils.logger import get_logger, log_event
from utils.config_utils import (
    inject_model_env, inject_env_from_config, resolve_model_config, load_config
)

logger = get_logger()

# Ensure cache directory exists
if not os.path.isdir(".cache"):
    os.makedirs(".cache")

def prepare_config_and_env(config_path: str, llm_override: dict = None) -> Dict[str, Any]:
    """Loads config, merges any LLM override, and injects all API/model keys into os.environ."""
    # 1. Load base config
    app_config = load_config(config_path)
    if not app_config:
        logger.error("Aborting: could not load configuration.")
        sys.exit(1)

    # 2. Merge CLI/frontend LLM overrides
    llm_override = llm_override or {}
    llm_section_name = f"{app_config.get('default_provider', 'openai').lower()}_llm_models"
    final_llm = app_config.get(llm_section_name, {}).copy()
    final_llm.update(llm_override)
    app_config['llm_models'] = final_llm

    # 3. Inject generic API/provider keys (all workers see them)
    inject_env_from_config(app_config)

    # 4. Inject all model API keys for current LLM models
    for agent_name, model_name in final_llm.items():
        if not model_name:
            continue
        try:
            model_cfg = resolve_model_config(model_name, app_config)
            inject_model_env(model_cfg)
            logger.info(f"Env preloaded for agent '{agent_name}' → model '{model_name}'")
            log_event("model_env_preloaded", {"agent": agent_name, "model": model_name})
        except Exception as e:
            logger.error(f"Failed to preload env for '{agent_name}': {e}")
            log_event("model_env_error", {"agent": agent_name, "error": str(e)})

    return app_config

def main():
    parser = argparse.ArgumentParser(description="FinRobot Annual Report Orchestration")
    parser.add_argument("company", type=str, help="Company name (e.g., 'Apple Inc.')")
    parser.add_argument("year", type=str, help="Year to analyze (e.g., '2023')")
    parser.add_argument("--config", type=str, default="langgraph_config.json", help="Path to the config JSON file.")
    parser.add_argument("--llm_models", type=str, default="{}", help="JSON string of agent→model mappings.")
    parser.add_argument("--report_type", type=str, default="kpi_bullet_insights", help="Type of report to generate")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    logger.info(f"Starting orchestration for {args.company} ({args.year})")
    log_event("orchestration_start", {
        "company": args.company,
        "year": args.year,
        "report_type": args.report_type,
        "verbose": args.verbose
    })

    # Parse frontend LLM override, if present
    try:
        frontend_llm = json.loads(args.llm_models)
        logger.info(f"Frontend LLM models: {frontend_llm}")
        log_event("frontend_llm_parsed", {"models": frontend_llm})
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing --llm_models JSON: {e}")
        frontend_llm = {}

    # 1. Load config, inject all env/API/model keys, merge overrides
    app_config = prepare_config_and_env(args.config, frontend_llm)

    # 2. (Optional) send config to frontend, or print for debug
    logger.info(f"Launching FinRobot orchestration for: {args.company} ({args.year})")
    if args.verbose:
        logger.info(f"App Config (verbose): {json.dumps(app_config, indent=2, ensure_ascii=False)}")

    # 3. Run orchestration (config passed as argument, never global)
    try:
        from workflows.orchestrator import run_orchestration

        run_orchestration(
            company=args.company,
            year=args.year,
            config=app_config,
            report_type=args.report_type,
            verbose=args.verbose
        )

        # Print ICAIF scores
        print("\n=== ICAIF Ratings ===")
        print(tabulate(eval["icaif_scores"].items(), headers=["Criterion", "Score"], tablefmt="github"))

        # Print pipeline matrix
        print("\n=== Pipeline Matrix ===")
        print(tabulate(eval["pipeline_matrix"], headers=["Agent", "Requests", "Avg Latency (ms)", "Errors"], tablefmt="github"))
        print("\n✅ Orchestration completed. Check logs and output files.\n")
        logger.info("Orchestration completed successfully.")
        log_event("orchestration_completed", {"company": args.company, "year": args.year})

    except ImportError as e:
        logger.error("Module 'workflows.orchestrator' not found.")
        log_event("orchestration_import_error", {"error": str(e)})
        sys.exit(1)
    except Exception as e:
        err_trace = traceback.format_exc()
        logger.error(f"Orchestration failed: {e}")
        log_event("orchestration_failed", {"error": str(e), "traceback": err_trace})
        sys.exit(1)


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    main()
