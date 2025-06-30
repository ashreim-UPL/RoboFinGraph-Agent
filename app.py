# app.py

import os
import json
import sys
import io
import subprocess
import traceback # For detailed error logging
from flask import Flask, jsonify, request, Response, stream_with_context
from typing import Dict, Any, List

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Configuration Paths (Define these as per your project structure) ---
# Assuming OAI_CONFIG_PATH is where your finrobot_config.json is located
OAI_CONFIG_PATH = "finrobot_config.json"
# Assuming ORCHESTRATOR_SCRIPT points to your main.py (the orchestrator entry point)
ORCHESTRATOR_SCRIPT = "main.py" # Or "workflows/orchestrator.py" if that's the direct script
APP_DIR = os.path.dirname(os.path.abspath(__file__)) # Directory where app.py resides
REPORT_DIR = os.path.join(APP_DIR, "report") # Directory for reports

# Ensure .cache directory exists
if not os.path.isdir(".cache"):
    os.makedirs(".cache")

# --- Logger (for consistency with previous code) ---
class AppLogger:
    def error(self, msg):
        print(f"ERROR: {msg}", file=sys.stderr)
    def info(self, msg):
        print(f"INFO: {msg}")
app.logger = AppLogger() # Assigning a simple logger for demonstration

# --- Core Configuration Loading Functions (from previous Canvas updates) ---

def _load_model_providers_from_file() -> Dict[str, Any]:
    """
    Loads the full configuration from finrobot_config.json.
    This function should return the entire dictionary loaded from the JSON file.
    """
    try:
        if not os.path.exists(OAI_CONFIG_PATH):
            app.logger.error(f"Configuration file not found at {OAI_CONFIG_PATH}. Please ensure it exists.")
            return {}
        with open(OAI_CONFIG_PATH, 'r') as f:
            full_config = json.load(f)
            if not isinstance(full_config, dict):
                app.logger.error(f"Invalid configuration format in {OAI_CONFIG_PATH}: Expected a top-level dictionary.")
                return {}
            return full_config
    except json.JSONDecodeError as e:
        app.logger.error(f"Error decoding JSON from {OAI_CONFIG_PATH}: {e}. Please check the file's JSON syntax.")
        return {}
    except Exception as e:
        app.logger.error(f"An unexpected error occurred while loading configuration from {OAI_CONFIG_PATH}: {e}")
        return {}


def get_available_models() -> Dict[str, Any]:
    """
    Identifies and returns models with non-empty API keys,
    along with the default agent-to-model assignments and a global default.
    This provides the frontend with initial configuration.
    """
    try:
        full_config = _load_model_providers_from_file()
        
        model_providers = full_config.get("model_providers", [])
        default_llm_models_from_config = full_config.get("llm_models", {})
        global_default_model_name = full_config.get("default_llm_model")

        filtered_models = {}
        idx = 0
        for provider_config in model_providers:
            provider_api_key = provider_config.get("api_key", "").strip()
            if provider_api_key:
                models_list = provider_config.get("models", [])
                for model_name in models_list:
                    if model_name and isinstance(model_name, str):
                        filtered_models[str(idx)] = model_name
                        idx += 1
            else:
                app.logger.info(f"Skipping provider '{provider_config.get('provider', 'Unknown')}' due to missing or empty API key.")
        
        return {
            "models": filtered_models,
            "default_agent_models": default_llm_models_from_config,
            "global_default_model": global_default_model_name # Can be None if not set
        }
    except Exception as e:
        app.logger.error(f"Failed to retrieve available models and defaults: {e}")
        return {
            "models": {},
            "default_agent_models": {},
            "global_default_model": None
        }

# --- API Key Registration (Call this on app startup) ---
# This function is crucial for setting environment variables for the orchestrator script.
# It needs resolve_model_config and inject_model_env if it's used directly here.
# For simplicity, if your orchestrator (main.py) handles its own env injection
# based on the passed config, you might not need this here.
# However, if any Flask-side operations depend on these, keep it.
# For now, let's assume it's still relevant.
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

def register_api_keys():
    """
    Registers API keys from both 'model_providers' and flat 'api_keys' in the config into os.environ.
    """
    try:
        full_config = _load_model_providers_from_file()
        model_providers = full_config.get("model_providers", [])

        for provider_config in model_providers:
            provider_type = provider_config.get("provider", "").lower()
            api_key = provider_config.get("api_key", "").strip()

            if not api_key:
                app.logger.info(f"No API key provided for provider '{provider_type}'. Skipping registration.")
                continue

            if provider_type == "openai":
                os.environ["OPENAI_API_KEY"] = api_key
                app.logger.info("Successfully registered OpenAI API key.")
                break
            elif provider_type == "together":
                os.environ["TOGETHER_API_KEY"] = api_key
                base_url = provider_config.get("base_url", "").strip()
                if base_url:
                    os.environ["TOGETHER_BASE_URL"] = base_url
                    app.logger.info("Successfully registered Together API key and base URL.")
                break
            else:
                app.logger.info(f"Encountered an unsupported provider type '{provider_type}'. API key not registered for this provider.")

        # --- NEW: Inject any loose API keys from the "api_keys" dict ---
        api_keys = full_config.get("api_keys", {})
        for key, value in api_keys.items():
            if value:
                os.environ[key] = value
                app.logger.info(f"Registered {key} from 'api_keys' section.")

    except Exception as e:
        app.logger.error(f"Error registering API keys: {str(e)}")


# Initialize on app start
# register_api_keys() # You might call this here, or rely on the orchestrator to do it.
                    # If orchestrator (main.py) handles all env injection, this might be optional.


# --- Routes ---
@app.route("/available_models")
def available_models_route(): # Renamed to avoid confusion with the function name
    """Return available LLM models and defaults from finrobot_config.json"""
    return jsonify(get_available_models())

@app.route("/report_files")
def report_files():
    """Return list of files in report directory"""
    # You'll need a get_report_files() function defined elsewhere or here
    # from flask import send_from_directory
    # import os
    # def get_report_files():
    #     if not os.path.exists(REPORT_DIR):
    #         return []
    #     return [f for f in os.listdir(REPORT_DIR) if os.path.isfile(os.path.join(REPORT_DIR, f))]
    return jsonify([]) # Placeholder if get_report_files is not defined yet

@app.route("/report/<path:filename>")
def reports(filename):
    """Serve report files"""
    from flask import send_from_directory
    return send_from_directory(REPORT_DIR, filename)

@app.route("/")
def index():
    from flask import render_template
    return render_template("finrobot.html")

@app.route("/stream")
def stream():
    """Stream analysis results via SSE"""
    company = request.args.get("company", "")
    year = request.args.get("year", "2024")
    # Updated to receive llm_models as JSON string
    llm_models_json_str = request.args.get("llm_models", "{}") 
    report_type = request.args.get("report_type", "kpi_bullet_insights")
    verbose = request.args.get("verbose", "false").lower() == "true"
    
    if not company:
        return Response("Company parameter required", status=400)
    
    # Check API key (consider if this check is still relevant here,
    # or if the orchestrator handles all key validation based on specific models)
    # If the orchestrator handles it, this check might be removed or generalized.
    if "OPENAI_API_KEY" not in os.environ and "TOGETHER_API_KEY" not in os.environ:
        app.logger.warning("Neither OPENAI_API_KEY nor TOGETHER_API_KEY found in environment. Orchestrator might fail if it relies on these.")
        # return Response("LLM API key not configured", status=500) # You might want to return an error here

    if not os.path.exists(ORCHESTRATOR_SCRIPT):
        return Response("Orchestrator script not found", status=500)

    # FIX: Pass company and year as positional arguments, and others as named arguments
    cmd = [
        "python", "-u", ORCHESTRATOR_SCRIPT,
        company, # Positional argument 1
        year,    # Positional argument 2
        "--report_type", report_type,
        "--llm_models", llm_models_json_str # Pass the JSON string directly
    ]
    if verbose:
        cmd.append("--verbose")
    
    # Pass environment with API key (already set by register_api_keys or expected by orchestrator)
    env = os.environ.copy()
    
    try:
        process = subprocess.Popen(
            cmd,
            cwd=APP_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            encoding='utf-8',
            errors="replace", 
            env=env
        )
    except Exception as e:
        error_msg = f"Subprocess failed to start: {str(e)}\n{traceback.format_exc()}"
        app.logger.error(error_msg)
        return Response(error_msg, status=500)
    
    def generate():
        def format_sse(data: dict) -> str:
            return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

        # Read stdout from the subprocess
        for line in iter(process.stdout.readline, ''):
            try:
                line = line.strip()
                if not line:
                    continue
                # Attempt to parse as JSON, otherwise treat as plain log
                if line.startswith("{") and line.endswith("}"):
                    log_event = json.loads(line)
                    yield format_sse(log_event)
                else:
                    yield format_sse({"event_type": "log", "data": {"message": line}})
            except json.JSONDecodeError:
                yield format_sse({"event_type": "log", "data": {"message": line}})
            except UnicodeDecodeError:
                safe_line = line.encode("utf-8", errors="replace").decode()
                yield format_sse({"event_type": "log", "data": {"message": safe_line}})
            except Exception as e:
                app.logger.error(f"Error processing stream line: {e} - Line: {line}")
                yield format_sse({"event_type": "log", "data": {"message": f"Error processing stream: {line}"}})


        stderr_output = process.stderr.read()
        if stderr_output:
            app.logger.error(f"Orchestrator stderr: {stderr_output}")
            yield format_sse({"event_type": "pipeline_error", "data": {"error": stderr_output}})

        process.wait()
        yield format_sse({"event_type": "pipeline_complete", "data": {"exit_code": process.returncode}})

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )

if __name__ == "__main__":
    # Call register_api_keys here if you want environment variables set when app.py starts
    # This is useful if Flask itself needs to make LLM calls or if the orchestrator
    # expects these to be pre-set.
    register_api_keys() # This will set the env vars based on finrobot_config.json

    print(f"Starting Flask app. Orchestrator script: {ORCHESTRATOR_SCRIPT}")
    app.run(debug=True, port=5000, use_reloader=False) # use_reloader=False is often good for subprocesses
