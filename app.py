# app.py

import os
import sys
import json
import io
import re
import subprocess
import urllib.parse
from datetime import datetime
from typing import Dict, Any, List

# --- Ensure local packages are importable ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from flask import Flask, jsonify, request, Response, stream_with_context, send_from_directory, render_template
from flask_cors import CORS

from utils.logger import get_logger, log_event
from utils.config_utils import resolve_model_config, inject_model_env, get_app_config, load_config, prepare_config_and_env 

logger = get_logger()

import logging
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

# Flask app init
FRONTEND_BUILD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'frontend', 'build'))
app = Flask(__name__, static_folder=FRONTEND_BUILD_DIR)
CORS(app, resources={r"/stream": {"origins": "*"}, r"/available_models": {"origins": "*"}})

# Centralized config loading for the Flask app.
try:
    initial_app_config = prepare_config_and_env(
        config_path=OAI_CONFIG_PATH,
        llm_override={}, 
        provider_override=None 
    )
    logger.info("Application configuration loaded and initialized successfully.")
except FileNotFoundError as e:
    logger.critical(f"Configuration file not found at {OAI_CONFIG_PATH}. Please ensure it exists. Error: {e}")
    sys.exit(1)
except Exception as e:
    logger.critical(f"Failed to load or prepare application configuration at startup: {e}")
    sys.exit(1)

@app.route('/doc/<path:filename>')
def serve_doc_file(filename):
    return send_from_directory('doc', filename)

@app.route('/available_models')
def available_models():
    try:
        config = get_app_config() 
        results = {}
        for prov_name, prov_data in config.get("providers", {}).items():
            agent_model_assignments = config.get(f"{prov_name}_llm_models", {})
            results[prov_name] = agent_model_assignments 

        logger.info(f"Available models: {json.dumps(results)}")
        return jsonify({"available_providers_llm": results}) 
    except RuntimeError as e:
        logger.error(f"Error accessing app configuration in /available_models: {e}")
        return jsonify({"error": "Application configuration not initialized."}), 500
    except Exception as e:
        logger.error(f"Failed to gather available models: {e}")
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500


@app.route("/report_files")
def report_files():
    try:
        files = [f for f in os.listdir(REPORT_DIR) if os.path.isfile(os.path.join(REPORT_DIR, f))]
        logger.info(f"report_files: {len(files)} files")
        return jsonify(files)
    except Exception as e:
        logger.error(f"report_files error: {e}")
        return jsonify([]), 500

@app.route("/")
def index():
    return render_template("robofingraph.html")

@app.route("/report/<path:filename>")
def serve_report(filename):
    return send_from_directory(REPORT_DIR, filename)


@app.route("/stream")
def stream():
    logger.info(f"STREAM endpoint called with args: {request.args.to_dict()}")
    log_event("stream_called", request.args.to_dict())      
    
    company = request.args.get("company")
    if not company:
        return jsonify({"error": "company required"}), 400
    year = request.args.get("year") or str(__import__('datetime').datetime.now().year)
    
    # Get parameters from the frontend URL
    report_type_param = request.args.get("report_type") 
    raw_llm_override_param = request.args.get("llm_models") 
    provider = request.args.get("provider")
    verbose_param = request.args.get("verbose", "false").lower() == "true" 

    if not os.path.exists(ORCHESTRATOR_SCRIPT):
        msg = f"Orchestrator not found: {ORCHESTRATOR_SCRIPT}"
        logger.error(msg)
        return jsonify({"error": msg}), 500

    # Build the base command
    cmd = [sys.executable, "-u", ORCHESTRATOR_SCRIPT, company, year]
            

    # Ensure these names match main.py's argparse EXACTLY
    if report_type_param: 
        cmd.extend(["--report_type", report_type_param])

    # Only add if it's explicitly present AND not an empty JSON string "{}"
    if raw_llm_override_param and raw_llm_override_param != "{}":
        llm_override_str = urllib.parse.unquote(raw_llm_override_param)
        cmd.extend(["--llm_models", llm_override_str])

    if provider:
        cmd.extend(["--provider", provider]) 

    if verbose_param: 
        cmd.append("--verbose")

    logger.debug(f"DEBUG: Command sent to main.py subprocess: {cmd}")

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
    log_dir = os.path.join(APP_DIR, "log")
    os.makedirs(log_dir, exist_ok=True)
    DEBUG_LOG_PATH = os.path.join(APP_DIR, "log", "flask_subprocess_debug.log")   
    def event_stream():
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as debug_log:
            for raw_item in proc.stdout:
                raw_string = str(raw_item).strip() 
                
                if not raw_string:
                    continue

                sys.stderr.write(f"RAW_RECEIVED_FINAL_PARSE: {raw_string!r}\n")
                debug_log.write(f"{datetime.now().isoformat()} | {raw_string!r}\n")
                debug_log.flush()

                parsed_successfully_as_structured_event = False
                
                log_prefix_pattern = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} \| (INFO|WARNING|ERROR) \| (.*)$")
                match = log_prefix_pattern.match(raw_string)
                
                potential_json_string = None
                if match:
                    potential_json_string = match.group(2).strip()
                
                if not potential_json_string or not (potential_json_string.startswith('{') and potential_json_string.endswith('}')):
                    json_start_index = raw_string.find('{')
                    json_end_index = raw_string.rfind('}')
                    if json_start_index != -1 and json_end_index != -1 and json_end_index > json_start_index:
                        potential_json_string = raw_string[json_start_index : json_end_index + 1].strip()
                    else:
                        potential_json_string = None


                if potential_json_string:
                    try:
                        data = json.loads(potential_json_string)
                        
                        if "event_type" in data and ("payload" in data or "data" in data):
                            event_name = data["event_type"]
                            payload_content = data.get("payload", data.get("data", data))
                            sys.stderr.write(f"DEBUG: YIELDING STRUCTURED EVENT: {event_name!r}\n")
                            # Yield a unified JSON for the frontend's .onmessage
                            yield f"data: {json.dumps({'event_type': event_name, 'payload': payload_content}, ensure_ascii=False)}\n\n"
                            parsed_successfully_as_structured_event = True
                    except json.JSONDecodeError:
                        sys.stderr.write(f"DEBUG: JSONDecodeError for: {potential_json_string!r}\n")
                        pass 

                if not parsed_successfully_as_structured_event:
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
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    app.run(debug=True, port=5000, use_reloader=False)