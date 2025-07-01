# app.py

import os
import sys
import json
import io
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
OAI_CONFIG_PATH = os.path.join(APP_DIR, "finrobot_config.json")
ORCHESTRATOR_SCRIPT = os.path.join(APP_DIR, "main.py")
REPORT_DIR = os.path.join(APP_DIR, "report")

# Ensure necessary directories
os.makedirs(os.path.join(APP_DIR, ".cache"), exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


def register_api_keys():
    """
    Load model_providers and flat api_keys from config and inject into env.
    """
    try:
        with open(OAI_CONFIG_PATH, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
    except Exception as e:
        logger.error(f"register_api_keys: failed to load config: {e}")
        log_event("register_api_keys_error", {"error": str(e)})
        return

    # Provider-based keys
    for prov in cfg.get("model_providers", []):
        key = prov.get("api_key", "").strip()
        api_type = prov.get("provider", "").lower()
        if not key:
            continue
        if api_type == "openai":
            os.environ["OPENAI_API_KEY"] = key
            logger.info("Injected OpenAI API key")
            log_event("api_key_injected", {"provider": "openai"})
        elif api_type == "together":
            os.environ["TOGETHER_API_KEY"] = key
            base = prov.get("base_url", "").strip()
            if base:
                os.environ["TOGETHER_BASE_URL"] = base
            logger.info("Injected Together API key")
            log_event("api_key_injected", {"provider": "together"})

    # Flat api_keys
    for k, v in cfg.get("api_keys", {}).items():
        if v:
            os.environ[k] = v
            logger.info(f"Injected API key: {k}")
            log_event("api_key_injected", {"key": k})


# Inject keys on import
register_api_keys()

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
    logger.info(f"available_models fetched: {len(filtered)}")
    log_event("available_models_fetched", {"count": len(filtered)})
    return jsonify({
        "models": filtered,
        "default_agent_models": defaults,
        "global_default_model": global_default
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


@app.route("/report/<path:filename>")
def serve_report(filename):
    return send_from_directory(REPORT_DIR, filename)


@app.route("/")
def index():
    return render_template("finrobot.html")


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
        for raw in proc.stdout:
            line = raw.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                event_name = data.get("event_type", "message")
                payload = data.get("data", data)
                yield f"event: {event_name}\n"
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            except json.JSONDecodeError:
                yield f"event: log\n"
                yield f"data: {json.dumps({"message": line}, ensure_ascii=False)}\n\n"
        # At completion
        proc.wait()
        yield "event: pipeline_complete\n"
        yield f"data: {json.dumps({"exit_code": proc.returncode}, ensure_ascii=False)}\n\n"

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