# finrobot/toolkit/global_API_toolkit.py
# Unified API toolkit with smart routing and SEC integration via inline logic

import os
import re
import socket
import requests
import logging
import time
import json
from typing import Annotated, Dict, Literal, Optional, Union, Any, List
from contextlib import contextmanager
from utils.logger import setup_logging, get_logger
from sec_api import ExtractorApi, QueryApi, RenderApi
from datetime import date

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
def get_api_config():
    return {
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
            "base_url": "https://container-finrobot.dc4gg5b1dous0.eu-west-3.cs.amazonlightsail.com/",
            "method": "POST",
            "api_key": "local"
        },
        "SEC": {
            "method": "CUSTOM",
            "api_key": os.getenv("SEC_API_KEY")
        }

    }

# Use everywhere:
API_CONFIG = get_api_config()
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
    config = get_api_config()[api_name]

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
            print("Sleeping for 1.5s to respect IndianMarket API rate limit.")
            time.sleep(1.5)
            response.raise_for_status()
            # print("reached api request: ", api_name, " response: ", response.json())
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "payload": payload}

    return execute() if api_name != "IndianMarket" else force_ipv4_context().__enter__() or execute()


def make_api_request2(
    api_name: Annotated[Literal["FMP", "IndianMarket", "LocalRAG"], "Target API name"],
    endpoint: Annotated[str, "API path like /query or /symbol/AAPL"],
    params: Annotated[Optional[Dict], "Payload or query parameters"] = None
) -> Annotated[dict, "JSON response or error"]:
    config = API_CONFIG.get(api_name)
    if not config or not config.get("api_key"):
        return {"error": f"API configuration or key is missing for '{api_name}'."}

    full_url = f"{config['base_url']}{endpoint}"
    headers = {}
    payload = params.copy() if isinstance(params, dict) else {}

    if api_name == "FMP":
        method = config.get("method", "GET").upper()
        if "auth_param" in config:
            payload[config["auth_param"]] = config["api_key"]
        elif "auth_header" in config:
            headers[config["auth_header"]] = config["api_key"]

        try:
            response = requests.request(method, full_url, json=payload if method == "POST" else None,
                                        params=payload if method == "GET" else None, headers=headers, timeout=20)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "payload": payload}

    elif api_name == "IndianMarket":
        if "auth_param" in config:
            payload[config["auth_param"]] = config["api_key"]
        elif "auth_header" in config:
            headers[config["auth_header"]] = config["api_key"]

        try:
            response = requests.get(full_url, params=payload, headers=headers, timeout=20)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "payload": payload}

    elif api_name == "LocalRAG":
        if "auth_param" in config:
            payload[config["auth_param"]] = config["api_key"]
        elif "auth_header" in config:
            headers[config["auth_header"]] = config["api_key"]

        try:
            response = requests.post(full_url, json=payload, headers=headers, timeout=20)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "payload": payload}

    else:
        return {"error": f"Unknown API name '{api_name}'."}

# ----------------------------------------------------------------------------
# ESTEBAN NEW METHOD
# ----------------------------------------------------------------------------

def make_api_request2(
    api_name: Annotated[Literal["FMP", "IndianMarket", "LocalRAG"], "Target API name"],
    endpoint: Annotated[str, "API path like /query or /symbol/AAPL"],
    params: Annotated[Optional[Dict], "Payload or query parameters"] = None
) -> Annotated[dict, "JSON response or error"]:
    config = API_CONFIG.get(api_name)
    if api_name == "IndianMarket":
        config["api_key"] = "sk-live-9CBc5SINdlUtl74sNSNTI7Yyc0aBYak5h46FRMzz"
    if not config or not config.get("api_key"):
        return {"error": f"API configuration or key is missing for '{api_name}'."}

    full_url = f"{config['base_url']}{endpoint}"
    headers = {}
    payload = params.copy() if isinstance(params, dict) else {}
    retries=5
    delay=3
    if api_name == "IndianMarket":
        if "auth_param" in config:
            payload[config["auth_param"]] = config["api_key"]
        elif "auth_header" in config:
            headers[config["auth_header"]] = config["api_key"]

        for attempt in range(retries):
            try:
                with force_ipv4_context():
                    response = requests.get(full_url, params=payload, headers=headers, timeout=20)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt < retries - 1:
                    print(f"API request failed (attempt {attempt+1}/{retries}): {e}. Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print(f"API request failed after {retries} attempts: {e}")
                    return {"error": str(e), "payload": payload}

    elif api_name == "LocalRAG":
        if "auth_param" in config:
            payload[config["auth_param"]] = config["api_key"]
        elif "auth_header" in config:
            headers[config["auth_header"]] = config["api_key"]

        try:
            response = requests.post(full_url, json=payload, headers=headers, timeout=20)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "payload": payload}

    else:
        return {"error": f"Unknown API name '{api_name}'."}
# ----------------------------------------------------------------------------
# SEC Toolkit Wrapper
# ----------------------------------------------------------------------------
# globals for the SEC client

def _init_sec_api() -> None:
    global query_api, extractor_api
    if query_api is None:
        from sec_api import QueryApi, ExtractorApi
        api_key = os.getenv("SEC_API_KEY")
        query_api     = QueryApi(api_key=api_key)
        extractor_api = ExtractorApi(api_key=api_key)

def get_10k_metadata(
    sec_ticker: str,
    start_date: str,
    end_date: str,
    filing_type: str = "10-K",
    limit: int = 1
) -> Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]:
    """
    Search for filings of type `filing_type` for `sec_ticker` between `start_date`
    and `end_date`. Return either the most recent one (if limit==1) or a list
    of up to `limit` filings. Returns None if nothing is found.
    """
    _init_sec_api() # Ensure this initializes extractor_api and query_api

    query = {
        "query": (
            f'ticker:"{sec_ticker}" AND formType:"{filing_type}" '
            f'AND filedAt:[{start_date} TO {end_date}]'
        ),
        "from": 0,
        "size": limit,
        "sort": [{"filedAt": {"order": "desc"}}],
    }

    # This is the actual call to your SEC API.
    # The previous example contained a mock here; this is the real one.
    resp = query_api.get_filings(query)
    filings = resp.get("filings", [])

    if not filings:
        return None

    # if the caller only wanted one back, give them a single dict
    if limit == 1:
        return filings[0]

    return filings
#---------------     ---------------------------------------------------
def get_10k_section(
    sec_ticker: str,
    fyear: str,
    sec_report_address: str,
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
    _init_sec_api() # Ensure this initializes extractor_api and query_api

    sec_str = str(section)
    valid = [str(i) for i in range(1, 16)] + ["1A", "1B", "7A", "9A", "9B"]
    if sec_str not in valid:
        raise ValueError(f"Invalid section '{sec_str}'. Must be one of {valid}")

    cache_dir = os.path.join("SEC_SECTION_CACHE")
    cache_file = os.path.join(cache_dir, f"{sec_ticker}_{fyear}_section_{sec_str}.txt")

    section_text = None # Initialize section_text

    # Try cache, else fetch and optionally cache
    if use_cache and os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            section_text = f.read()
    else:
        # Use the provided sec_report_address for the first attempt
        # This will be the problem URL (e.g., CIK browse page) if the pipeline provides it.
        # Ensure it's cleaned if it has "Link: " prefix
        initial_report_address = sec_report_address.lstrip("Link: ").split()[0]
        
        try:
            # First attempt: fetch the desired section from the initial URL.
            # This is where the 'filing type not supported' error often originates.
            logger.info(f"Attempting initial fetch for Section {section} from {initial_report_address}")
            section_text = extractor_api.get_section(initial_report_address, section, "text")

        except Exception as e:
            # Log the error for diagnostics
            logger.warning(f"Initial fetch failed for Section {section} at {initial_report_address}: {e}. Initiating fallback to get precise filing URL.")

            year = int(fyear)
            # Define search range for metadata, general for any ticker
            start_date_range = f"{year-1}-01-01" 
            end_date_range = f"{year}-12-31"

            # Fallback: re-query the SEC index for the correct 10-K filing using the general get_10k_metadata
            # THIS CALL USES YOUR EXISTING, UNMODIFIED get_10k_metadata
            metadata = get_10k_metadata(
                sec_ticker = sec_ticker,
                filing_type = "10-K",
                start_date = start_date_range,
                end_date = end_date_range,
                limit = 1
            )
            
            # Print statements should generally be replaced by logging in production,
            # but I'm leaving them as per your original request's debugging nature.
            # These are for the *fallback path* only.
            # print("ticker_sec :", sec_ticker,"from: ", start_date_range, "to: ", end_date_range)
            # if metadata:
            #     print("metadata filing: ", metadata.get("linkToFilingDetails"))
            #     print("metadata html", metadata.get("linkToHtml"))
            # input("sce_ticket test") # Remove this line unless you explicitly need to pause execution for manual check

            doc_url = None
            if metadata:
                # Prioritize direct HTML/TXT links from metadata
                doc_url = (
                    metadata.get("linkToHtml")
                    or metadata.get("linkToTxt")
                    or metadata.get("linkToFilingDetails")
                )

            if metadata and doc_url:
                logger.info(f"Fallback successful: Retrying Section {section} from resolved URL: {doc_url}")
                try:
                    # Second attempt: using the correctly resolved direct document URL
                    section_text = extractor_api.get_section(doc_url, section, "text")
                except Exception as inner_e:
                    # If even the direct URL fails (e.g., network, or actual issue with SEC API),
                    # raise a clear RuntimeError.
                    raise RuntimeError(
                        f"Failed to fetch Section {section} for {sec_ticker} in {fyear} "
                        f"even after resolving direct URL {doc_url}: {inner_e}"
                    ) from inner_e
            else:
                # If metadata isn't found or a document URL can't be resolved,
                # then we can't proceed.
                raise RuntimeError(
                    f"No valid 10-K filing found for {sec_ticker} in {fyear}; "
                    f"cannot fetch Section {section}. Metadata or doc_url missing after fallback."
                )
 
        # After either the initial attempt or the fallback, ensure section_text is set.
        if section_text is None:
            # This should ideally be caught by the RuntimeErrors above, but as a final safeguard.
            raise RuntimeError(
                f"Section {section} could not be retrieved for {sec_ticker} in {fyear} after all attempts."
            )

        # Save to cache for future runs (only if fetched, not if from cache)
        if use_cache:
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as f:
                f.write(section_text)

    # Always save to save_path (if specified), after ensuring section_text is available
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(section_text)

    return {"text": section_text}
#-------                  ------------------------------------

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
    load_files_from_directory,
    make_api_request2
]
