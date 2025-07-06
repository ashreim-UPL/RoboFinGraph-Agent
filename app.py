# app.py

import os
import sys
import json
import io
import re
import subprocess
import traceback
import urllib.parse
from typing import Dict, Any, List

# --- Ensure local packages are importable ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from flask import Flask, jsonify, request, Response, stream_with_context, send_from_directory, render_template
from flask_cors import CORS

from utils.logger import get_logger, log_event
from utils.config_utils import resolve_model_config, inject_model_env

logger = get_logger()

import logging, sys
console = logging.StreamHandler(sys.stdout)
console.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
get_logger().addHandler(console)

# Paths
OAI_CONFIG_PATH = os.path.join(APP_DIR, "langgraph_config.json")
ORCHESTRATOR_SCRIPT = os.path.join(APP_DIR, "main.py")
REPORT_DIR = os.path.join(APP_DIR, "report")

# Ensure necessary directories
os.makedirs(os.path.join(APP_DIR, ".cache"), exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


def load_llm_models(config_path: str) -> dict:
    """
    Loads only LLM model mappings from the config JSON.
    Returns a dict:
      {
        "openai_llm_models": {...},
        "mixtral_llm_models": {...},
        "qwen_llm_models": {...}
      }
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        models = {
            "openai_llm_models": cfg.get("openai_llm_models", {}),
            "mixtral_llm_models": cfg.get("mixtral_llm_models", {}),
            "qwen_llm_models": cfg.get("qwen_llm_models", {}),
        }
        logger.info("Loaded LLM model mappings from config")
        log_event("llm_models_loaded", {"models": list(models.keys())})
        return models
    except Exception as e:
        logger.error(f"Failed to load LLM model mappings: {e}")
        log_event("llm_models_load_error", {"error": str(e)})
        return {}

# Flask app init
app = Flask(__name__)
CORS(app, resources={r"/stream": {"origins": "*"}, r"/available_models": {"origins": "*"}})


def _load_full_config() -> Dict[str, Any]:
    try:
        with open(OAI_CONFIG_PATH, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict):
            raise ValueError("Config root not dict")
        return cfg
    except Exception as e:
        logger.error(f"_load_full_config error: {e}")
        log_event("config_load_error", {"error": str(e)})
        return {}


@app.route("/available_models")
def available_models():
    cfg = _load_full_config()
    providers = cfg.get("model_providers", [])
    defaults = cfg.get("llm_models", {})
    global_default = cfg.get("default_llm_model")
    filtered = {}
    idx = 0
    for p in providers:
        if p.get("api_key", "").strip():
            for m in p.get("models", []):
                if isinstance(m, str):
                    filtered[str(idx)] = m
                    idx += 1
    
    # NEW: Call load_llm_models to get the structured provider data
    available_providers_llm = load_llm_models(OAI_CONFIG_PATH) #

    logger.info(f"available_models fetched: {len(filtered)} models, {len(available_providers_llm)} providers")
    log_event("available_models_fetched", {"count": len(filtered), "providers_count": len(available_providers_llm)})
    
    return jsonify({
        "models": filtered,
        "default_agent_models": defaults,
        "global_default_model": global_default,
        "available_providers_llm": available_providers_llm # # Add this to the response
    })


@app.route("/report_files")
def report_files():
    try:
        files = [f for f in os.listdir(REPORT_DIR) if os.path.isfile(os.path.join(REPORT_DIR, f))]
        logger.info(f"report_files: {len(files)} files")
        return jsonify(files)
    except Exception as e:
        logger.error(f"report_files error: {e}")
        return jsonify([]), 500

FRONTEND_BUILD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'frontend', 'build'))
app = Flask(__name__, static_folder=FRONTEND_BUILD_DIR)

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

"""@app.route("/")
def index():
    return render_template("finrobot.html")"""

@app.route("/report/<path:filename>")
def serve_report(filename):
    return send_from_directory(REPORT_DIR, filename)


@app.route("/stream")
def stream():
    logger.info(f"STREAM endpoint called with args: {request.args.to_dict()}")
    log_event("stream_called", request.args.to_dict())      
    # Required params
    company = request.args.get("company")
    if not company:
        return jsonify({"error": "company required"}), 400
    year = request.args.get("year") or str(__import__('datetime').datetime.now().year)
    # Decode percent-encoded JSON
    raw_llm = request.args.get("llm_models", "{}")
    llm_str = urllib.parse.unquote(raw_llm)
    report_type = request.args.get("report_type", "kpi_bullet_insights")
    verbose = request.args.get("verbose", "false").lower() == "true"

    if not os.path.exists(ORCHESTRATOR_SCRIPT):
        msg = f"Orchestrator not found: {ORCHESTRATOR_SCRIPT}"
        logger.error(msg)
        return jsonify({"error": msg}), 500

    # Build subprocess cmd
    cmd = [sys.executable, "-u", ORCHESTRATOR_SCRIPT,
            company, year,
            "--report_type", report_type,
            "--llm_models", llm_str]
    if verbose:
        cmd.append("--verbose")

    logger.info(f"Launching orchestrator subprocess with cmd: {cmd}")
    log_event("orchestrator_cmd", {"cmd": cmd})

    env = os.environ.copy()
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=APP_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env
        )
    except Exception as e:
        logger.error(f"stream start error: {e}")
        log_event("stream_start_error", {"error": str(e)})
        return jsonify({"error": str(e)}), 500

    def event_stream():
        # Read lines from subprocess
        for raw_item in proc.stdout: # Use raw_item to represent the direct yield from proc.stdout
            # Ensure raw_item is converted to string immediately.
            # Use 'replace' for errors to prevent UnicodeEncodeError in this debug output,
            # though the primary fix should be in logger.py and environment.
            raw_string = str(raw_item).strip() 
            
            if not raw_string:
                continue

            # --- Debugging line to see every raw string coming in ---
            import sys
            sys.stderr.write(f"RAW_RECEIVED_FINAL_PARSE: {raw_string!r}\n")
            # --- End Debugging line ---

            parsed_successfully_as_structured_event = False
            
            # Attempt to parse the line as a logger-prefixed JSON event.
            # This regex is specifically designed to strip the logger's INFO/WARNING/ERROR prefix
            # and capture the potential JSON string that follows.
            # Using re.match for start-of-string matching.
            log_prefix_pattern = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} \| (INFO|WARNING|ERROR) \| (.*)$")
            match = log_prefix_pattern.match(raw_string)
            
            potential_json_string = None
            if match:
                potential_json_string = match.group(2).strip() # Group 2 captures everything after the level and final '| '

            # If it didn't match the logger prefix, or the part after the prefix doesn't start with '{',
            # then try to find the first '{' for cases where the format might vary.
            if not potential_json_string or not potential_json_string.startswith('{'):
                json_start_index = raw_string.find('{')
                if json_start_index != -1:
                    potential_json_string = raw_string[json_start_index:].strip()
                else:
                    potential_json_string = None # No JSON part found at all

            if potential_json_string and potential_json_string.startswith('{') and potential_json_string.endswith('}'):
                try:
                    data = json.loads(potential_json_string)
                    
                    # Ensure it's a structured event generated by your log_event function
                    if "event_type" in data and ("payload" in data or "data" in data):
                        event_name = data["event_type"]
                        payload_content = data.get("payload", data.get("data", data))
                        
                        sys.stderr.write(f"DEBUG: YIELDING STRUCTURED EVENT: {event_name!r}\n")
                        yield f"event: {event_name}\n"
                        yield f"data: {json.dumps(payload_content, ensure_ascii=False)}\n\n"
                        parsed_successfully_as_structured_event = True
                except json.JSONDecodeError:
                    sys.stderr.write(f"DEBUG: JSONDecodeError for: {potential_json_string!r}\n")
                    pass # Not valid JSON, or not a structured event, or malformed JSON

            if not parsed_successfully_as_structured_event:
                # If no structured event was successfully parsed, yield the original raw_string as a generic log message.
                sys.stderr.write(f"DEBUG: YIELDING FALLBACK LOG: {raw_string!r}\n")
                yield f"event: log\n"
                yield f"data: {json.dumps({'message': raw_string}, ensure_ascii=False)}\n\n"

    return Response(
        stream_with_context(event_stream()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )

if __name__ == "__main__":
    logger.info("Starting Flask app")
    log_event("flask_start", {})
    # Ensure UTF-8 stdout
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    app.run(debug=True, port=5000, use_reloader=False)