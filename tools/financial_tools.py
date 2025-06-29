# tools/financial_tools.py

from langchain.tools import tool
from typing import Dict, Any, Optional

from tools.global_API_toolkit import make_api_request
from tools.company_resolver import identify_company_and_region

# === Company Resolver (Async wrapped call) ===
@tool
def resolve_company_name(company_query: str) -> Dict[str, Any]:
    """Fix company typos and resolve official name, region, and tickers."""
    import asyncio
    return asyncio.run(identify_company_and_region(company_query))


# === API Call Wrappers ===
@tool
def get_stock_data(ticker: str) -> Dict[str, Any]:
    """Fetch stock quote and historical metrics for a company."""
    return make_api_request("FMP", f"/quote/{ticker}", {})

@tool
def get_filing_date(ticker: str) -> Dict[str, Any]:
    """Fetch the most recent SEC filing date."""
    return make_api_request("FMP", f"/sec_filings/{ticker}", {"limit": 1})

@tool
def fetch_peers(ticker: str) -> Dict[str, Any]:
    """Get competitor companies for a given ticker."""
    return make_api_request("FMP", f"/stock_peers?symbol={ticker}", {})

