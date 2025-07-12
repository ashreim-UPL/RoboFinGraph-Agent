# main.py

import argparse
import json
import os
import sys
import io
import traceback
from typing import Dict, Any

from utils.logger import get_logger, log_event
from utils.config_utils import prepare_config_and_env

logger = get_logger()

# Ensure cache directory exists
if not os.path.isdir(".cache"):
    os.makedirs(".cache")

def main():
    parser = argparse.ArgumentParser(description="RoboFinGraph Annual Report Orchestration")
    parser.add_argument("company",    type=str, help="Company name (e.g., 'Apple Inc.')")
    parser.add_argument("year",       type=str, help="Year to analyze (e.g., '2023')")
    parser.add_argument("--config",   type=str, default="langgraph_config.json",
                        help="Path to the config JSON file.")
    parser.add_argument("--provider", type=str,
                        help="Which LLM provider to load (must match a key under `providers` in config)")
    parser.add_argument("--llm_models", type=str, default="{}",
                        help="(future) per-agent overrides — only used if you need to patch individual models")
    parser.add_argument("--report_type", type=str, default="kpi_bullet_insights",
                        help="Type of report to generate")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    # parse per-agent override JSON (optional)
    try:
        frontend_llm = json.loads(args.llm_models)
    except json.JSONDecodeError:
        frontend_llm = {}

    # 1) load config & inject ALL env, using provider_override + any per-agent overrides
    app_config = prepare_config_and_env(
        config_path       = args.config,
        llm_override      = frontend_llm,
        provider_override = args.provider
    )

    logger.info(f"Starting orchestration for {args.company} ({args.year})")
    log_event("orchestration_start", {
        "company":    args.company,
        "year":       args.year,
        "report_type":args.report_type,
        "config":     args.config,
        "provider":   args.provider,
        "llm_models": args.llm_models,
        "verbose":    args.verbose
    })

    if args.verbose:
        logger.info(f"App Config (verbose): {json.dumps(app_config, indent=2)}")

    # 2) run the pipeline
    try:
        from workflows.orchestrator import run_orchestration

        run_orchestration(
            company     = args.company,
            year        = args.year,
            config      = app_config,
            report_type = args.report_type,
            verbose     = args.verbose
        )

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
    # ensure UTF-8 output on Windows
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    main()