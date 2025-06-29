# finrobot/toolkit/global_API_toolkit.py
# Unified API toolkit with smart routing and SEC integration via inline logic

import os
import re
import socket
import requests
import logging
from typing import Annotated, Dict, Literal, Optional
from contextlib import contextmanager
from utils.logger import setup_logging, get_logger
from sec_api import ExtractorApi, QueryApi, RenderApi

# ----------------------------------------------------------------------------
# Initialize Logging and Environment
# ----------------------------------------------------------------------------
setup_logging()
logger = get_logger("finrobot.toolkit")


BASE_SAVE_DIR = os.getenv("OUTPUT_DIR", "report")
os.makedirs(BASE_SAVE_DIR, exist_ok=True)

# ----------------------------------------------------------------------------
# API Configuration (dynamic extensions supported)
# ----------------------------------------------------------------------------
API_CONFIG = {
    "FMP": {
        "base_url": "https://financialmodelingprep.com/stable",
        "method": "GET",
        "auth_param": "apikey",
        "api_key": os.getenv("FMP_API_KEY")
    },
    "IndianMarket": {
        "base_url": "https://stock.indianapi.in",
        "method": "GET",
        "auth_header": "x-api-key",
        "api_key": os.getenv("INDIAN_API_KEY")
    },
    "LocalRAG": {
        "base_url": "http://127.0.0.1:8000",
        "method": "POST",
        "api_key": "local"
    },
    "SEC": {
        "method": "CUSTOM",
        "api_key": os.getenv("SEC_API_KEY")
    }
}

# ----------------------------------------------------------------------------
# Init SEC API Clients
# ----------------------------------------------------------------------------
SEC_API_KEY = os.getenv("SEC_API_KEY")
extractor_api = ExtractorApi(SEC_API_KEY) if SEC_API_KEY else None
query_api = QueryApi(SEC_API_KEY) if SEC_API_KEY else None
render_api = RenderApi(SEC_API_KEY) if SEC_API_KEY else None
PDF_GENERATOR_API = "https://api.sec-api.io/filing-reader"
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")

# ----------------------------------------------------------------------------
# Utility Helpers
# ----------------------------------------------------------------------------
@contextmanager
def force_ipv4_context():
    original = socket.getaddrinfo
    try:
        socket.getaddrinfo = lambda *args, **kwargs: [info for info in original(*args, **kwargs) if info[0] == socket.AF_INET]
        yield
    finally:
        socket.getaddrinfo = original

def normalize_path(path: str) -> str:
    return os.path.join(BASE_SAVE_DIR, os.path.basename(path))

# ----------------------------------------------------------------------------
# Smart API Request Handler (FMP, India, RAG, etc.)
# ----------------------------------------------------------------------------
def make_api_request(
    api_name: Annotated[Literal["FMP", "IndianMarket", "LocalRAG"], "Target API name"],
    endpoint: Annotated[str, "API path like /query or /symbol/AAPL"],
    params: Annotated[Optional[Dict], "Payload or query parameters"] = None
) -> Annotated[dict, "JSON response or error"]:
    config = API_CONFIG.get(api_name)
    if not config or not config.get("api_key"):
        return {"error": f"API configuration or key is missing for '{api_name}'."}

    full_url = f"{config['base_url']}{endpoint}"
    method = config.get("method", "GET").upper()
    headers = {}
    payload = params.copy() if isinstance(params, dict) else {}

    if api_name == "IndianMarket":
        if "statement" not in payload and "stats" in payload:
            hint = str(payload["stats"]).lower()
            for k, v in {"income": "yoy_results", "earning": "yoy_results", "cash": "cashflow", "balance": "balancesheet", "quarter": "quarter_results"}.items():
                if k in hint:
                    payload["statement"] = v
                    break

    if api_name == "LocalRAG":
        q = payload.get("question", "").lower()
        if "year" not in payload:
            match = re.search(r"\\b(19|20)\\d{2}\\b", q)
            if match:
                payload["year"] = int(match.group(0))
        if "section" not in payload:
            for k, s in {"risk": "risk", "governance": "governance", "board": "governance", "revenue": "Management", "overview": "Management"}.items():
                if k in q:
                    payload["section"] = s
                    break

    if api_name != "LocalRAG":
        if "auth_param" in config:
            payload[config["auth_param"]] = config["api_key"]
        elif "auth_header" in config:
            headers[config["auth_header"]] = config["api_key"]

    def execute():
        try:
            response = requests.request(method, full_url, json=payload if method == "POST" else None,
                                        params=payload if method == "GET" else None, headers=headers, timeout=20)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "payload": payload}

    return execute() if api_name != "IndianMarket" else force_ipv4_context().__enter__() or execute()

# ----------------------------------------------------------------------------
# SEC Toolkit Wrapper
# ----------------------------------------------------------------------------
def _init_sec_api():
    global query_api, extractor_api
    if query_api is None:
        from sec_api import QueryApi, ExtractorApi
        api_key = API_CONFIG["SEC"]["api_key"]
        query_api = QueryApi(api_key=api_key)
        extractor_api = ExtractorApi(api_key=api_key)
        
def call_sec_utility(
    action: Annotated[str, "Action to perform, e.g., get_10k_section"],
    params: Annotated[dict, "Parameter dict for the SEC action"]
) -> dict:
    _init_sec_api()
    try:
        if action == "get_10k_metadata":
            start_date = params['start_date']
            end_date = params['end_date']
            query = {
                "query": f'ticker:"{params["ticker"]}" AND formType:"10-K" AND filedAt:[{start_date} TO {end_date}]',
                "from": 0, "size": 1, "sort": [{"filedAt": {"order": "desc"}}]
            }
            filings_response = query_api.get_filings(query)
            filings = filings_response.get('filings', [])
            if not filings:
                return {"error": f"No 10-K filings found for {params['ticker']} in range {start_date} to {end_date}"}
            return filings[0]

        elif action == "get_10k_section":
            meta = call_sec_utility("get_10k_metadata", {
                "ticker": params["ticker_symbol"],
                "start_date": f"{params['fyear']}-01-01",
                "end_date": f"{int(params['fyear']) + 1}-06-30"
            })
            if meta.get("error"):
                return meta
            report_url = meta.get("linkToHtml") or meta.get("linkToFilingDetails")
            if not report_url:
                return {"error": "Filing metadata found, but it contains no URL."}
            
            return {"text": extractor_api.get_section(report_url, str(params["section"]), "text")}
        else:
            return {"error": f"Unsupported SEC action: {action}"}
    except Exception as e:
        logging.error(f"SEC utility action '{action}' failed: {e}")
        return {"error": str(e)}

def save_to_file(data: str, file_path: str) -> str:
    """Saves string data to a file, creating directories if needed."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(data)
        return f"Successfully saved to {os.path.abspath(file_path)}"
    except Exception as e:
        logging.error(f"Failed to save file to {file_path}: {e}")
        raise RuntimeError(f"Failed to save file: {e}")

# ----------------------------------------------------------------------------
# File I/O
# ----------------------------------------------------------------------------
def save_to_file(data: str, file_path: str) -> str:
    """
    Saves string data to a file at the specified file_path,
    creating parent directories if they don't exist.
    """
    try:
        # --- MODIFICATION START ---
        # REMOVED the call to the aggressive normalize_path() helper.
        # We will now use the file_path directly as provided.
        # This ensures that subdirectories like 'sec_filings' are respected.
        
        # Ensure the parent directory exists.
        parent_dir = os.path.dirname(file_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        
        # Write the data to the file.
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(data)
            
        # --- MODIFICATION END ---
            
        return f"Successfully saved to {os.path.abspath(file_path)}"
    except Exception as e:
        logging.error(f"Failed to save file to {file_path}: {e}")
        raise RuntimeError(f"Failed to save file: {e}")

def load_files_from_directory(
    directory_path: Annotated[str, "Directory with .txt files"]
) -> Annotated[Dict[str, str], "Map of filename to content"]:
    """
    Loads all .txt files from a specified directory and returns their content
    in a dictionary mapping filename to content.
    """
    if not os.path.isdir(directory_path): 
        raise ValueError(f"Directory '{directory_path}' does not exist.")
        
    out = {}
    for fname in os.listdir(directory_path):
        if fname.endswith(".txt"):
            fpath = os.path.join(directory_path, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    out[fname] = f.read()
            except Exception as e:
                out[fname] = f"[ERROR reading file: {e}]"
    return out

# ----------------------------------------------------------------------------
# Exported Toolkit
# ----------------------------------------------------------------------------
UNIVERSAL_ANALYST_TOOLKIT = [
    make_api_request,
    call_sec_utility,
    save_to_file,
    load_files_from_directory
]
