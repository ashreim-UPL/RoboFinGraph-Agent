from openai import OpenAI
import os
import json
import logging
from typing import Dict, Any, Optional
import re

from utils.logger import get_logger, log_event

# Ensure this import path is correct for your project structure.
from tools.global_API_toolkit import make_api_request

def process_company_data(company_name: str, year: str) -> Dict[str, Any]:
    """
    Combines LLM information extraction with FMP-based company name verification.

    Args:
        company_name (str): The name of the company to query.
        year (str): The year for which filing data is requested.

    Returns:
        Dict[str, Any]: A dictionary containing LLM results, FMP verification results,
                        validation status, accuracy, and a message.
    """

    # Prepare the result dict
    results: Dict[str, Any] = {
        "llm_result": None,
        "message": "Processing initiated.",
        "tokens_sent": 0,
        "tokens_generated": 0,
        "cost_llm": 0.0,
    }

    # 2. Check API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        err = "OPENAI_API_KEY is not set. Please configure it."
        log_event("company_data_error", {"company": company_name, "year": year, "error": err})
        raise ValueError(err)

    client = OpenAI(api_key=api_key)

    # --- Phase 1: LLM Information Extraction ---
    try:
        prompt = f"""
        Return the following as a JSON:

        - official_name
        - company_name
        - segments
        - industry
        - stock_market_names
        - fmp_ticker
        - sec_ticker
        - sec_report_address
        - yfinance_ticker
        - region
        - filing_date for {year}
        - currency
        - current share price
        - peers

        Company: {company_name}
        Do not add any text or other explanation. Keep your response as JSON output only.
        """

        # Manually specify pricing for gpt-4o-search-preview (USD per 1 000 tokens)
        PRICE_PROMPT_PER_1K     = 0.002  # e.g. $0.002 per 1 000 prompt tokens
        PRICE_COMPLETION_PER_1K = 0.002  # e.g. $0.002 per 1 000 completion tokens

        llm_response = client.chat.completions.create(
            model="gpt-4o-mini-search-preview",
            messages=[{"role": "user", "content": prompt}],
            timeout=20,
        )

        # 5. Parse JSON out of any markdown fences
        raw = llm_response.choices[0].message.content
        m = re.search(r"```(?:json)?(.*?)```", raw, re.DOTALL)
        json_str = m.group(1).strip() if m else raw.strip()
        llm_data = json.loads(json_str)
        results["llm_result"] = llm_data

        # if region shows North Amerca chnage to USA
        if llm_data.get("region") == "North America":
            llm_data["region"] = "USA"

        # 6. Extract tokens
        usage            = getattr(llm_response, "usage", {}) or {}
        prompt_toks = getattr(usage, "prompt_tokens",     0)
        completion_toks = getattr(usage, "completion_tokens", 0)
        results["tokens_sent"]      = prompt_toks
        results["tokens_generated"] = completion_toks
        # 7. Compute cost using our hard-coded rates
        cost = (
            (prompt_toks     * PRICE_PROMPT_PER_1K) +
            (completion_toks * PRICE_COMPLETION_PER_1K)
        ) / 1_000 

        results["cost_llm"] = cost

        # 8. Log structured success
        log_event(
            "company_data_extracted",
            {
                "company": company_name,
                "year": year,
                "tokens_sent": prompt_toks,
                "tokens_generated": completion_toks,
                "cost_llm": cost
            }
        )

    except Exception as e:
        msg = f"Error during LLM data extraction: {e}"
        log_event("company_data_error", {"company": company_name, "year": year, "error": str(e)})
        results["message"]        = msg
        # leave tokens & cost at their zero defaults

    return results
