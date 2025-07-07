from openai import OpenAI
import os
import json
import asyncio
import logging
from typing import Dict, Any, Optional
import re

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

    # Suppress warnings if running this as a standalone snippet without full project structure
    logging.basicConfig(level=logging.INFO)
    company_resolver_logger = logging.getLogger("CompanyResolver")

    # OpenAI API client initialization
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        # A positive and assertive way to handle missing API key
        raise ValueError("OPENAI_API_KEY environment variable is not set. Please ensure it is configured for seamless operation.")
    client = OpenAI(api_key=api_key)

    results = {
        "llm_result": None,
        "message": "Processing initiated."
    }

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

        llm_response = client.chat.completions.create(
            model="gpt-4o-search-preview-2025-03-11",
            messages=[{"role": "user", "content": prompt}]
        )
        def clean_llm_json(llm_output_str):
            # Remove markdown code block fences (``` or ```json ... ```)
            code_block_pattern = r"```(?:json)?(.*?)```"
            match = re.search(code_block_pattern, llm_output_str, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
            else:
                # No code block detected, use raw string
                json_str = llm_output_str.strip()
            # Try parsing
            try:
                return json.loads(json_str)
            except Exception as e:
                raise ValueError(f"Could not parse LLM output as JSON. Error: {e}\nRaw output:\n{llm_output_str}")

        llm_output_str = llm_response.choices[0].message.content
        llm_data  = clean_llm_json(llm_output_str)

        usage = getattr(llm_response, "usage", None)
        results["tokens_sent"] = usage.prompt_tokens if usage and hasattr(usage, "prompt_tokens") else 0
        results["tokens_generated"] = usage.completion_tokens if usage and hasattr(usage, "completion_tokens") else 0
        results["cost_llm"] = 0.0
        try:
            results["llm_result"] = llm_data
            company_resolver_logger.info(f"LLM successfully extracted data for {company_name}.")
        except Exception as e:
            results["message"] = f"LLM output could not be set: {e}"
            results["success_score"] = 0
            company_resolver_logger.error(f"LLM produced invalid JSON for {company_name}: {llm_data}")
            return results

    except Exception as e:
        results["message"] = f"Error during LLM data extraction: {e}"
        results["accuracy_score"] = 0
        company_resolver_logger.critical(f"Critical error during LLM call for {company_name}: {e}", exc_info=True)
        return results
    return results

# --- Example Usage ---
def main():
    # Ensure OPENAI_API_KEY is set in your environment variables before running
    # e.g., in PowerShell: $env:OPENAI_API_KEY="sk-proj-YOUR-KEY-HERE"

    print(f"\n--- Processing Company: 'TLSA' ---")
    tesla_result = process_company_data("Tesla", "2024")
    print(json.dumps(tesla_result, indent=2))

    print(f"\n--- Processing Company: 'Starlink' ---")
    # For Starlink, LLM might return a ticker, but FMP won't have a public profile for it,
    # leading to FMP_Verification_Unavailable or Hallucination_Detected if LLM gives wrong ticker.
    starlink_result = process_company_data("Starlink", "2024")
    print(json.dumps(starlink_result, indent=2))

    print(f"\n--- Processing Company: 'QuantumLeap Innovations'")
    # LLM might not even give a ticker for a fictional company, or FMP won't find it.
    quantum_result = process_company_data("QuantumLeap Innovations", "2024")
    print(json.dumps(quantum_result, indent=2))
    
    print(f"\n--- Processing Company: 'Abble' ---")
    apple_result = process_company_data("Apple Inc.", "2024")
    print(json.dumps(apple_result, indent=2))

if __name__ == "__main__":
    main()