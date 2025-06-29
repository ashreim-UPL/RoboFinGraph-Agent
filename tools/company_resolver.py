# D:/dev/FinRobot/finrobot/company_resolver.py

import logging
import json
import re
import os
import sys
import asyncio # Added for running synchronous code in an async context
from typing import Optional, Dict, Any

import aiohttp

# Import the centralized API request handler
# This assumes 'finrobot' is a source root in your project structure.
from tools.global_API_toolkit import make_api_request

# --- HARDCODED CONFIGURATION VALUES ---
YAHOO_FINANCE_SEARCH_URL = "https://query1.finance.yahoo.com/v1/finance/search"
DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

# --- Module-level logger ---
company_resolver_logger = logging.getLogger("CompanyResolver")

# NOTE: The 'force_ipv4' global monkey-patch has been removed as requested by the
# original source code's comment. Network-wide settings should be managed at the
# application entry point or via a dedicated networking utility module.
# The 'global_API_toolkit' provides a context manager for this, which is a better practice.


# --- Company Identification and Classification ---
async def identify_company_and_region(company_query: str) -> dict:
    """
    Identifies a company's official name, region (IN, US, or Unknown),
    various tickers/identifiers, AND its top 3 main competitors using Yahoo Finance
    search and the global_API_toolkit for competitor data.
    """
    if not isinstance(company_query, str) or not company_query.strip():
        company_resolver_logger.error("Invalid 'company_query': Must be a non-empty string.")
        return {"validation_error": "Invalid 'company_query': Must be a non-empty string."}

    headers = {'User-Agent': DEFAULT_USER_AGENT}
    search_url = f"{YAHOO_FINANCE_SEARCH_URL}?q={company_query.strip()}"

    company_info = {
        "company_query": company_query,
        "company_details": {
            "official_name": None,
            "region": "Unknown",
            "identifiers": {},
            "message": "",
            "competitors": []
        },
        "region": "Unknown"
    }

    # --- Step 1: Primary Identification via Yahoo Finance ---
    try:
        async with aiohttp.ClientSession(headers=headers) as session:
            # Note: The 'force_ipv4' patch was here. If Yahoo calls fail on certain
            # networks, consider a more robust async-compatible network configuration.
            async with session.get(search_url, timeout=10) as resp:
                resp.raise_for_status()
                data = await resp.json()

        if not data.get("quotes"):
            company_info["tool_error"] = "No matching company found for primary identification."
            company_resolver_logger.warning(f"No matching company found for '{company_query}' via Yahoo Finance search.")
            return company_info

        top = data["quotes"][0]
        official_name = top.get("longname") or top.get("shortname") or top.get("symbol")
        exchange = top.get("exchange")
        symbol = top.get("symbol")

        indian_exchanges = ("NSE", "BSE", "IND")
        us_exchanges = ("NMS", "NYQ", "PCX", "NAS", "ASE", "NYSE", "NASDAQ")

        region = "Unknown"
        if symbol and (symbol.endswith(".NS") or symbol.endswith(".BO")):
            region = "IN"
        elif exchange in indian_exchanges:
            region = "IN"
        elif exchange in us_exchanges:
            region = "US"

        company_info["company_details"].update({
            "official_name": official_name,
            "region": region,
            "identifiers": {
                "ticker": symbol,
                "yfinance_ticker": symbol,
                "fmp_ticker": symbol.split('.')[0] if symbol else None,
                "finnhub_symbol": symbol.split('.')[0] if symbol else None,
                "sec_cik": None,
                "indian_stock_name": official_name if region == "IN" else None,
                "indian_stock_id": None
            },
            "message": f"Company identified as {official_name} ({symbol} on {exchange}, Region: {region})"
        })
        company_info["region"] = region

    except aiohttp.ClientError as e:
        company_resolver_logger.error(f"Network error during company identification for '{company_query}': {e}", exc_info=True)
        company_info["tool_error"] = f"Network or API communication error: {str(e)}"
        return company_info
    except Exception as e:
        company_resolver_logger.error(f"Unexpected error during identification for '{company_query}': {e}", exc_info=True)
        company_info["tool_error"] = f"An unexpected error occurred during identification: {str(e)}"
        return company_info

    # --- Step 2: Competitor Lookup using global_API_toolkit ---
    fmp_ticker = company_info["company_details"]["identifiers"].get("fmp_ticker")
    competitor_list = []
    
    if fmp_ticker:
        try:
            company_resolver_logger.info(f"🕵️‍♀️ Searching for competitors for {fmp_ticker}...")
            
            response = await asyncio.to_thread(
                make_api_request,
                api_name="FMP",
                endpoint="/stock-peers",
                params={"symbol": fmp_ticker}
            )
  
            # --- MODIFICATION START: New, robust response handling ---

            # First, check if the response is a dictionary containing an error.
            if isinstance(response, dict) and response.get("error"):
                error_msg = response.get("error")
                company_resolver_logger.error(
                    f"Error from Universal Analyst Toolkit during competitor lookup for {fmp_ticker}: {error_msg}"
                )
            
            # Next, check if the response is a non-empty list, which indicates success.
            elif isinstance(response, list) and response:
                peers_data = response[0] # The successful response is a list containing one dictionary
                peers = peers_data.get("peers", [])
                
                if peers and isinstance(peers, list):
                    competitor_list = [peer.strip() for peer in peers if isinstance(peer, str)][:3]
                    
                    # Clean the list to not include the company itself
                    primary_name_lower = (company_info['company_details']['official_name'] or "").lower()
                    primary_ticker_lower = (fmp_ticker or "").lower()
                    
                    competitor_list = [
                        comp for comp in competitor_list
                        if primary_name_lower not in comp.lower() and primary_ticker_lower not in comp.lower()
                    ]
                    company_resolver_logger.info(f"✅ Found competitors via Universal Analyst Toolkit: {competitor_list}")
                else:
                    company_resolver_logger.warning(f"No peers found in the FMP response for {fmp_ticker}.")
            
            # Handle any other unexpected cases (e.g., empty list, wrong data type).
            else:
                company_resolver_logger.warning(
                    f"Received an empty or unexpected response during competitor lookup for {fmp_ticker}."
                )

            # --- MODIFICATION END ---

        except Exception as e:
            company_resolver_logger.error(f"Failed to execute competitor lookup for {fmp_ticker} via toolkit: {e}", exc_info=True)

    company_info["company_details"]["competitors"] = competitor_list
    company_info["competitors"] = competitor_list  # Maintain for backward compatibility

    return company_info