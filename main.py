# main.py
import argparse
import json
import os, sys, io
from typing import Dict, Any, List
# This tool is not provided, assuming it loads a JSON file into a dict.
from tools.file_utils import load_config_into_environ 
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if not os.path.isdir(".cache"):
    os.makedirs(".cache")

def resolve_model_config(model_name: str, model_providers: List[Dict[str, Any]]) -> Dict[str, Any]:
    for provider in model_providers:
        if model_name in provider["models"]:
            config = {
                "model": model_name,
                "api_key": provider["api_key"],
                "api_type": provider["provider"]
            }
            if "base_url" in provider:
                config["base_url"] = provider["base_url"]
            return config
    raise ValueError(f"Model '{model_name}' not found in model_providers.")

def inject_model_env(model_config: Dict[str, Any]):
    api_key = model_config.get("api_key") or ""
    if model_config["api_type"] == "openai":
        os.environ["OPENAI_API_KEY"] = api_key
    elif model_config["api_type"] == "together":
        os.environ["TOGETHER_API_KEY"] = api_key
        os.environ["TOGETHER_BASE_URL"] = model_config.get("base_url", "")

def main():
    parser = argparse.ArgumentParser(description="FinRobot Annual Report Orchestration")
    parser.add_argument("company", type=str, help="Enter the company name (e.g., 'Apple Inc.')")
    parser.add_argument("year", type=str, help="Enter the year to analyze (e.g., 2023)")
    parser.add_argument("--config", type=str, default="finrobot_config.json", help="Path to the config JSON file.")
    args = parser.parse_args()

    # Load and export full config
    config = load_config_into_environ(args.config)

    # Preload default LLM model environment (leader)
    leader_model = config.get("llm_models", {}).get("leader")
    if leader_model:
        leader_config = resolve_model_config(leader_model, config.get("model_providers", []))
        inject_model_env(leader_config)

    print(f"\n📊 Launching FinRobot orchestration for: {args.company} ({args.year})\n")
    
    # MODIFICATION: Pass the 'config' object to the orchestration function
    from workflows.orchestrator import run_orchestration
    run_orchestration(args.company, args.year, config)
    
    print("\n✅ Orchestration completed. Check logs and output files.\n")


if __name__ == "__main__":
    main()