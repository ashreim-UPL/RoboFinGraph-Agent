# finrobot/toolkit/global_API_toolkit.py
# Unified API toolkit with smart routing and SEC integration via inline logic

import os
import re
import socket
import requests
import logging
import json
from typing import Annotated, Dict, Literal, Optional, Union, Any
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
# globals for the SEC client
_query_api: QueryApi = None
_extractor_api: ExtractorApi = None

def _init_sec_api() -> None:
    global _query_api, _extractor_api
    if _query_api is None:
        from sec_api import QueryApi, ExtractorApi
        api_key = API_CONFIG["SEC"]["api_key"]
        _query_api     = QueryApi(api_key=api_key)
        _extractor_api = ExtractorApi(api_key=api_key)

def get_10k_metadata(
    ticker_symbol: str,
    start_date: str,
    end_date: str,
) -> Optional[Dict[str, Any]]:
    """
    Search for 10-K filings for `ticker_symbol` between `start_date` and `end_date`,
    and return the metadata of the most recent one (or None if none found).
    """
    _init_sec_api() 
    query = {
        "query": (
            f'ticker:"{ticker_symbol}" AND formType:"10-K" '
            f'AND filedAt:[{start_date} TO {end_date}]'
        ),
        "from": 0,
        "size": 1,
        "sort": [{"filedAt": {"order": "desc"}}],
    }
    resp = _query_api.get_filings(query)
    filings = resp.get("filings", [])
    return filings[0] if filings else None

def get_10k_section(
    ticker_symbol: str,
    fyear: str,
    section: Union[int, str],
    save_path: Optional[str] = None,
    use_cache: bool = True
) -> Dict[str, str]:
    """
    Extracts a given section (e.g. 1, "1A", 7) from the most
    recent 10-K for ticker_symbol in fiscal year fyear.
    Caches to SEC_SECTION_CACHE/<ticker>_<year>_section_<sec>.txt
    and optionally writes to save_path.
    Returns {"text": <section_body>}.
    """
    _init_sec_api()

    # normalize section code
    sec_str = str(section)
    valid = [str(i) for i in range(1,16)] + ["1A","1B","7A","9A","9B"]
    if sec_str not in valid:
        raise ValueError(f"Invalid section '{sec_str}'. Must be one of {valid}")

    # build cache path
    cache_dir  = os.path.join("SEC_SECTION_CACHE")
    cache_file = os.path.join(cache_dir, f"{ticker_symbol}_{fyear}_section_{sec_str}.txt")

    # return cached if available
    if use_cache and os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            return {"text": f.read()}

    # 1) fetch metadata for latest 10-K in that year
    query = {
        "query": f'ticker:"{ticker_symbol}" AND formType:"10-K" AND filedAt:[{fyear}-01-01 TO {fyear}-12-31]',
        "from": 0, "size": 1, "sort": [{"filedAt": {"order": "desc"}}]
    }
    meta_resp = _query_api.get_filings(query)
    filings   = meta_resp.get("filings", [])
    if not filings:
        raise RuntimeError(f"No 10-K filings found for {ticker_symbol} in {fyear}")
    meta = filings[0]

    # 2) pick the best URL
    report_url = meta.get("linkToTxt") or meta.get("linkToHtml")
    if not report_url:
        raise RuntimeError(f"No .txt/.htm URL in SEC metadata for {ticker_symbol}/{fyear}")

    # 3) call extractor
    section_text = _extractor_api.get_section(report_url, sec_str, "text")
    if not section_text:
        raise RuntimeError(f"Extractor returned empty for section {sec_str}")

    # 4) save to cache
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            f.write(section_text)

    # 5) optionally save to user path
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(section_text)

    return {"text": section_text}

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
    get_10k_metadata,
    get_10k_section,
    save_to_file,
    load_files_from_directory
]
