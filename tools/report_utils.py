# tools/report_utils.py

import json
import re
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, Any, Annotated, Tuple
from collections import defaultdict
import os
import logging


# Import the core API calling functions from your toolkit
from .global_API_toolkit import make_api_request, call_sec_utility, save_to_file
from .charting import *

# This new version REPLACES the previous get_key_data function.
def get_key_data(
    ticker_symbol: Annotated[str, "ticker_symbol"],
    filing_date: Annotated[str | datetime, "filing date of the financial report"],
    save_path: str,
    fyear: str, 
    region: str 
) -> str:
    """
    Returns key financial data for the given ticker by calling the global_api_toolkit,
    ensuring all calls follow the standardized format.
    """
    print(f"Fetching key data for {ticker_symbol}...")
    
    # --- Date conversion and range calculation (no changes here) ---
    if isinstance(filing_date, str):
        filing_date_obj = datetime.strptime(filing_date, "%Y-%m-%d")
    else:
        filing_date_obj = filing_date

    start_date = (filing_date_obj - timedelta(weeks=52)).strftime("%Y-%m-%d")
    end_date = filing_date_obj.strftime("%Y-%m-%d")
    
    # --- API Calls via global_api_toolkit (Refactored for Consistency) ---
    
    # Get Historical Data
    hist_data_json = make_api_request("FMP", "/historical-price-eod/full", {"symbol": ticker_symbol, "from": start_date, "to": end_date})

    hist = pd.DataFrame() # Default empty DataFrame
    if isinstance(hist_data_json, dict) and 'historical' in hist_data_json:
        data = hist_data_json['historical']
    elif isinstance(hist_data_json, list):
        data = hist_data_json
    else:
        data = []

    if data:
        hist = pd.DataFrame(data)
        if not hist.empty and 'date' in hist.columns:
            hist['date'] = pd.to_datetime(hist['date'])
            hist = hist.set_index('date')
    # Get Company Profile
    profile_response = make_api_request("FMP", "/profile", {"symbol": ticker_symbol})
    profile = profile_response[0] if profile_response and isinstance(profile_response, list) else {}
    currency = profile.get("currency")

    # Get Analyst Ratings (using corrected endpoint and safe access)
    rating_response = make_api_request("FMP", "/ratings-historical", {"symbol": ticker_symbol, "limit": 1})
    rating = rating_response[0].get('rating', 'N/A') if rating_response and isinstance(rating_response, list) else 'N/A'

    # Get Ratios for BVPS (Book Value Per Share)
    ratios_response = make_api_request("FMP", "/ratios", {"symbol": ticker_symbol, "period": "annual", "limit": 1})
    bvps_raw = ratios_response[0].get('bookValuePerShare', 0) if ratios_response and isinstance(ratios_response, list) else 0
    bvps_formatted = f"{bvps_raw:.2f}"

    # Get Market Cap
    market_cap_response = make_api_request("FMP", "/market-capitalization", {"symbol": ticker_symbol})
    market_cap_value = market_cap_response[0].get('marketCap', 0) if market_cap_response and isinstance(market_cap_response, list) else 0
    
    # --- Data Processing (no changes here) ---
    if hist.empty:
        close_price, fifty_two_week_low, fifty_two_week_high = 0.0, 0.0, 0.0
    else:
        # Ensure index is datetime for min/max operations
        hist.index = pd.to_datetime(hist.index)
        close_price = hist["close"].iloc[-1]
        fifty_two_week_low = hist["low"].min()
        fifty_two_week_high = hist["high"].max()

    # --- Assemble result (no changes here) ---
    suffix = f" ({currency})" if currency else ""
    key_data_dict = {
        "Rating": rating,
        "Currency": currency or "N/A",
        f"Closing Price{suffix}": f"{close_price:.2f}",
        f"Market Cap{suffix} (Millions)": f"{market_cap_value / 1e6:.2f}",
        f"52 Week Price Range{suffix}": f"{fifty_two_week_low:.2f} - {fifty_two_week_high:.2f}",
        f"BVPS{suffix}": bvps_formatted,
    }
    
    # Save the final dictionary to a file
    return save_to_file(json.dumps(key_data_dict, indent=2), save_path)

# --- I. Foundational Company Data Functions ---
def get_company_profile(ticker: str, save_path: str, **kwargs) -> str:
    """
    Retrieves company profile and description exclusively from FMP.
    """
    print(f"Fetching company profile for {ticker} from FMP...")
    # MODIFIED: Removed all region-specific and SEC-related logic
    content = make_api_request("FMP", "/profile", {"symbol": ticker})
    return save_to_file(json.dumps(content, indent=2), save_path)

def get_competitor_analysis(ticker: str, save_path: str, **kwargs) -> str:
    """
    Retrieves a list of competitors from FMP.
    """
    print(f"Fetching competitors for {ticker}...")
    content = make_api_request("FMP", "/stock-peers", {"symbol": ticker})
    return save_to_file(json.dumps(content, indent=2), save_path)



# --- II. Core Financial Statement Functions ---
def get_financial_statement(ticker: str, statement_type: str, save_path: str, **kwargs) -> str:
    """
    Retrieves a core financial statement (income, balance, cash-flow) from FMP.
    """
    print(f"Fetching {statement_type} for {ticker} from FMP...")
    # MODIFIED: Removed all SEC-related logic for a direct FMP call
    statement_map = {
        "income_statement": "income-statement",
        "balance_sheet": "balance-sheet-statement",
        "cash_flow_statement": "cash-flow-statement"
    }
    endpoint = f"/{statement_map.get(statement_type)}"
    content = make_api_request("FMP", endpoint, {"symbol": ticker, "period": "annual"})
    return save_to_file(json.dumps(content, indent=2), save_path)

# --- III. Key Performance Metrics Functions ---

def get_key_metrics(ticker: str, region: str, save_path: str) -> str:
    """
    Retrieves key financial metrics from FMP.
    """
    print(f"Fetching key metrics for {ticker}...")
    content = make_api_request("FMP", "/key-metrics", {"symbol": ticker})
    return save_to_file(json.dumps(content, indent=2), save_path)

def get_historical_prices(ticker: str, region: str, save_path: str) -> str:
    """
    Retrieves historical stock prices from FMP.
    """
    print(f"Fetching historical prices for {ticker}...")
    content = make_api_request("FMP", "/historical-price-eod/full?", {"symbol": ticker})
    return save_to_file(json.dumps(content, indent=2), save_path)


def get_sec_10k_sections(ticker: str, fyear: str, save_path: str, **kwargs) -> str:
    """
    Fetches sections 1, 1A, and 7 from the latest 10-K filing for a US company
    and saves each section to a separate text file.
    """
    print(f"Fetching SEC 10-K sections 1, 1A, and 7 for {ticker}...")
    
    os.makedirs(save_path, exist_ok=True)
    sections_to_fetch = ["1", "1A", "7"]
    created_files = []

    for section_id in sections_to_fetch:
        try:
            # The 'content_dict' variable holds the dictionary returned from the API call
            content_dict = call_sec_utility(
                "get_10k_section",
                {"ticker_symbol": ticker, "fyear": fyear, "section": section_id}
            )

            # --- MODIFICATION START: Check the dictionary and extract the text ---
            # Check if the call was successful and returned a dictionary with the 'text' key
            if content_dict and isinstance(content_dict, dict) and "text" in content_dict:
                text_to_save = content_dict["text"] # Extract the string value
                
                file_name = f"sec_10k_section_{section_id.lower()}.txt"
                full_path = os.path.join(save_path, file_name)
                
                # Pass the extracted text string to be saved
                save_to_file(text_to_save, full_path)
                created_files.append(full_path)
                logging.info(f"Successfully saved {full_path}")
            else:
                # Log the error message returned from the utility
                error_info = content_dict.get('error', 'Unknown error') if isinstance(content_dict, dict) else 'Invalid response'
                logging.warning(f"Could not retrieve Section {section_id} for {ticker}. Response: {error_info}")
            # --- MODIFICATION END ---

        except Exception as e:
            logging.error(f"An error occurred while fetching Section {section_id} for {ticker}: {e}", exc_info=True)

    return f"SEC section data saving process completed for {ticker}. Files created: {len(created_files)}"

# --- IV. Chart and Report Generation Functions ---

def generate_share_performance_chart(ticker: str, fyear: str, save_path: str, **kwargs) -> str:
    return get_share_performance(ticker, f"{fyear}-12-31", save_path)

def generate_pe_eps_chart(ticker: str, fyear: str, save_path: str, **kwargs) -> str:
    return get_pe_eps_performance(ticker, f"{fyear}-12-31", save_path)

def get_financial_metrics(
    ticker_symbol: str,
    save_path: str,
    years: int = 5,
    **kwargs
) -> str:

    """
    Returns a DataFrame containing financial metrics for the last N years for a given ticker symbol,
    with years as columns and metrics as rows.
    """
    all_metrics_by_year = defaultdict(dict)
    params = {"limit": years + 1}
    # Fetch each endpoint via make_api_request (using /stable)
    income_data = make_api_request("FMP", f"/income-statement?symbol={ticker_symbol}", params)
    key_metrics_data = make_api_request("FMP", f"/key-metrics?symbol={ticker_symbol}", params)
    ratios_data = make_api_request("FMP", f"/ratios?symbol={ticker_symbol}", params)
    cashflow_data = make_api_request("FMP", f"/cash-flow-statement?symbol={ticker_symbol}", params)
    profile_data = make_api_request("FMP", f"/profile?symbol={ticker_symbol}")
    # Handle error cases up front
    if any('error' in x for x in [income_data, key_metrics_data, ratios_data, cashflow_data]):
        return pd.DataFrame(), "USD", ticker_symbol.upper()
    if not all(isinstance(x, list) for x in [income_data, key_metrics_data, ratios_data, cashflow_data]):
        return pd.DataFrame(), "USD", ticker_symbol.upper()

    # --- Process metrics by year ---
    for i in range(min(years, len(income_data))):
        if i < len(key_metrics_data) and i < len(ratios_data) and i < len(cashflow_data):
            income = income_data[i]
            key_metrics = key_metrics_data[i]
            ratios = ratios_data[i]
            cashflow = cashflow_data[i]
            free_cash_flow = cashflow.get("freeCashFlow", 0)

            revenue = income.get("revenue", 0)
            gross_profit = income.get("grossProfit", 0)
            net_income = income.get("netIncome", 1e-9) or 1e-9

            metrics = {
                "Revenue": round(revenue / 1e6),
                "Gross Profit": round(gross_profit / 1e6),
                "Gross Margin": round((gross_profit / revenue) if revenue else 0, 2),
                "EBITDA": round(income.get("ebitda", 0) / 1e6),
                "EBITDA Margin": round(ratios.get("ebitdaMargin", 0), 2),
                "FCF": round(free_cash_flow / 1e6),
                "FCF Conversion": round((free_cash_flow / net_income), 2),
                "ROIC": f"{round(key_metrics.get('returnOnInvestedCapital', 0) * 100, 1)}%",
                "EV/EBITDA": round(key_metrics.get("evToEBITDA", 0), 2),
                "PE Ratio": round(ratios.get("priceToEarningsRatio", 0), 2),
                "PB Ratio": round(ratios.get("priceToBookRatio", 0), 2),
                "CFO": round(cashflow.get("operatingCashFlow", 0) / 1e6),
            }

            # Revenue growth (YoY)
            revenue_growth_val = "N/A"
            if i + 1 < len(income_data):
                prev_revenue = income_data[i + 1].get("revenue", 0)
                if prev_revenue:
                    growth = ((revenue - prev_revenue) / prev_revenue) * 100
                    revenue_growth_val = f"{round(growth, 1)}%"
            metrics["Revenue Growth"] = revenue_growth_val

            year = income.get("date", str(pd.Timestamp.now().year - i))[:4]
            all_metrics_by_year[year].update(metrics)

    # --- Create DataFrame ---
    df = pd.DataFrame(all_metrics_by_year)
    kpi_order = [
        "Revenue", "Revenue Growth", "Gross Profit", "Gross Margin", 
        "EBITDA", "EBITDA Margin", "FCF", "FCF Conversion", "ROIC",
        "EV/EBITDA", "PE Ratio", "PB Ratio", "CFO"
    ]
    df = df.reindex(kpi_order).dropna(how='all')

    # Currency and company name (fallbacks)
    currency = income_data[0].get("reportedCurrency", "USD")
    name = ticker_symbol.upper()
    if isinstance(profile_data, list) and profile_data:
        profile = profile_data[0]
        currency = profile.get("currency", currency)
        name = profile.get("companyName", name)

    # --- SAVE TO FILE if save_path is given ---
    if save_path:
        try:
            # Save as JSON for consistency (easy reloading)
            df_json = df.to_dict(orient="index")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump({
                    "currency": currency,
                    "company_name": name,
                    "metrics": df_json
                }, f, indent=2)
            logging.info(f"Financial metrics saved to {save_path}")
        except Exception as e:
            logging.error(f"Failed to save financial metrics: {e}")

    return df.sort_index(axis=1), currency, name