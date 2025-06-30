# main.py
import argparse
import json
import os, sys, io
import traceback
from typing import Dict, Any, List

# Placeholder for load_config_into_environ if it's not a separate tool
def load_config_into_environ(config_path: str) -> Dict[str, Any]:
    """
    Loads the full configuration from a JSON file.
    """
    try:
        with open(config_path, 'r') as f:
            full_config = json.load(f)
            if not isinstance(full_config, dict):
                print(f"Error: Invalid configuration format in {config_path}: Expected a top-level dictionary.", file=sys.stderr)
                return {}
            return full_config
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}", file=sys.stderr)
        return {}
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {config_path}: {e}", file=sys.stderr)
        return {}
    except Exception as e:
        print(f"An unexpected error occurred loading config: {e}", file=sys.stderr)
        return {}

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if not os.path.isdir(".cache"):
    os.makedirs(".cache")

class SimpleLogger:
    def error(self, msg):
        print(f"ERROR: {msg}", file=sys.stderr)
    def info(self, msg):
        print(f"INFO: {msg}")
app = type('obj', (object,), {'logger': SimpleLogger()})()


def resolve_model_config(model_name: str, model_providers: List[Dict[str, Any]]) -> Dict[str, Any]:
    for provider in model_providers:
        if "models" in provider and model_name in provider["models"]:
            config = {
                "model": model_name,
                "api_key": provider.get("api_key", ""),
                "api_type": provider.get("provider", "")
            }
            if "base_url" in provider:
                config["base_url"] = provider["base_url"]
            return config
    raise ValueError(f"Model '{model_name}' not found in model_providers.")

def inject_model_env(model_config: Dict[str, Any]):
    api_key = model_config.get("api_key") or ""
    if model_config["api_type"] == "openai":
        os.environ["OPENAI_API_KEY"] = api_key
        app.logger.info(f"Injected OPENAI_API_KEY for model: {model_config.get('model')}")
    elif model_config["api_type"] == "together":
        os.environ["TOGETHER_API_KEY"] = api_key
        os.environ["TOGETHER_BASE_URL"] = model_config.get("base_url", "")
        app.logger.info(f"Injected TOGETHER_API_KEY and BASE_URL for model: {model_config.get('model')}")
    else:
        app.logger.info(f"No environment variable injection rule for API type: {model_config['api_type']}")


def main():
    parser = argparse.ArgumentParser(description="FinRobot Annual Report Orchestration")
    parser.add_argument("company", type=str, help="Enter the company name (e.g., 'Apple Inc.')")
    parser.add_argument("year", type=str, help="Enter the year to analyze (e.g., 2023)")
    parser.add_argument("--config", type=str, default="finrobot_config.json", help="Path to the config JSON file.")
    parser.add_argument("--llm_models", type=str, default="{}",
                        help="JSON string of agent-to-model mappings (e.g., '{\"leader\": \"model_x\", \"data_cot\": \"model_y\"}')")
    parser.add_argument("--report_type", type=str, default="kpi_bullet_insights", help="Type of report to generate")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Load full config from file
    app_config = load_config_into_environ(args.config)

    # --- FIX: Inject ALL API keys from the 'api_keys' section into os.environ ---
    api_keys_from_config = app_config.get("api_keys", {})
    for key_name, key_value in api_keys_from_config.items():
        if key_name and key_value: # Only set if key_name and key_value are not empty
            os.environ[key_name] = key_value
            app.logger.info(f"Injected API key from config: {key_name}")
        else:
            app.logger.warning(f"Skipping empty or missing API key in config: {key_name}")
    # --- END FIX ---

    # Parse llm_models from the command line argument (from frontend)
    frontend_llm_models_config = {}
    try:
        frontend_llm_models_config = json.loads(args.llm_models)
        app.logger.info(f"Frontend LLM models received: {frontend_llm_models_config}")
    except json.JSONDecodeError as e:
        app.logger.error(f"Error parsing --llm_models argument JSON: {e}. Using empty config.")
    except Exception as e:
        app.logger.error(f"Unexpected error with --llm_models argument: {e}. Using empty config.")

    # Merge or override 'llm_models' in the main config with frontend selections
    final_llm_models_config = app_config.get("llm_models", {}).copy()
    final_llm_models_config.update(frontend_llm_models_config)
    app_config["llm_models"] = final_llm_models_config
    app.logger.info(f"Final LLM models configuration for orchestration: {app_config['llm_models']}")

    # Preload environment for the 'leader' model based on the final config
    # This is still here, but now all other API keys are also injected.
    leader_model_name = app_config.get("llm_models", {}).get("leader")
    if leader_model_name:
        try:
            leader_config = resolve_model_config(leader_model_name, app_config.get("model_providers", []))
            inject_model_env(leader_config)
            app.logger.info(f"Environment preloaded for leader model: {leader_model_name}")
        except ValueError as e:
            app.logger.error(f"Could not resolve or inject environment for leader model '{leader_model_name}': {e}")
        except Exception as e:
            app.logger.error(f"Error preloading environment for leader model: {e}")

    print(f"\n📊 Launching FinRobot orchestration for: {args.company} ({args.year})\n")
    
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
    except ImportError:
        app.logger.error("Error: 'workflows.orchestrator' module not found. Please ensure it's in your Python path.")
        print("\n❌ Orchestration failed: Missing orchestrator module.\n")
    except Exception as e:
        error_msg = f"Orchestration failed: {str(e)}\n{traceback.format_exc()}"
        app.logger.error(error_msg)
        print(f"\n❌ Orchestration failed: {str(e)}\n")


if __name__ == "__main__":
    main()
