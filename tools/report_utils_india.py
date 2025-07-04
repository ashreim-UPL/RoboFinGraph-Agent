import json
import re
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, Any, Annotated, Tuple
from collections import defaultdict
import os
import logging


# Import the core API calling functions from your toolkit
from .global_API_toolkit import make_api_request2, save_to_file
from .charting import *
from .rag_api_utils import get_annual_report_section, perform_similarity_search


def get_cached_stock_data(ticker, force_refresh=False):
    cash_dir = "cash_data"
    os.makedirs(cash_dir, exist_ok=True)
    fname = f"{cash_dir}/{ticker}_stock.json"
    if os.path.exists(fname) and not force_refresh:
        with open(fname, "r") as f:
            return json.load(f)
    else:
        data = make_api_request2("IndianMarket", "/stock", {"name": ticker})
        with open(fname, "w") as f:
            json.dump(data, f, indent=2)
        return data

def get_key_data_india(ticker: str, save_path: str) -> str:

    r= get_cached_stock_data(ticker)
    # r = make_api_request2("IndianMarket", "/stock", {"name": ticker})
    print(f"API response for {ticker}:\n{r}")

    if not isinstance(r, dict) or not r:
        print("API returned no or invalid data")
        return save_to_file(json.dumps({"error": "No data"}), save_path)

    profile = r.get("companyProfile", {})
    curr = r.get("currency", "INR")
    curp = r.get("currentPrice", {}) or {}
    price = curp.get("NSE") or curp.get("BSE") or 0

    # Defensive: keyMetrics
    key_metrics = r.get('keyMetrics') or {}
    priceandvolume = key_metrics.get('priceandVolume') or []
    lo = hi = None
    for data in priceandvolume:
        k = str(data.get('key', '')).lower()
        v = data.get('value')
        try:
            if k == '52weeklow':
                lo = float(v)
            elif k == '52weekhigh':
                hi = float(v)
        except (TypeError, ValueError):
            continue

    # BVPS
    try:
        bvps_val = float(key_metrics.get("bookValuePerShare", 0))
    except Exception:
        bvps_val = 0.0

    # Market Cap
    market_cap = 0.0
    peer_company_list = profile.get('peerCompanyList', [])
    if peer_company_list and isinstance(peer_company_list, list):
        first = peer_company_list[0] if isinstance(peer_company_list[0], dict) else {}
        market_cap = float(first.get('marketCap', 0.0))

    # Analyst view
    av = r.get("analystView", [])
    if not isinstance(av, list):
        av = []
    ratings = [item for item in av if item.get('ratingName') not in ('Total', '') and 'ratingValue' in item and 'numberOfAnalystsLatest' in item]
    if ratings:
        total_analysts = sum(int(item['numberOfAnalystsLatest']) for item in ratings)
        weighted = sum(int(item['ratingValue']) * int(item['numberOfAnalystsLatest']) for item in ratings) / total_analysts if total_analysts else 0
        def map_score_to_text(val):
            if val < 1.5: return "Strong Buy"
            if val < 2.5: return "Buy"
            if val < 3.5: return "Hold"
            if val < 4.5: return "Sell"
            return "Strong Sell"
        rating = f"{map_score_to_text(weighted)} (score: {weighted:.2f})" if total_analysts else "N/A"
    else:
        rating = "N/A"

    key_data = {
        "Rating": rating,
        "Currency": curr,
        "Closing Price": price,
        "Market Cap (Millions)": f"{market_cap/1e6:.2f}" if market_cap else "N/A",
        "52 Week Price Range": f"{lo:.2f} - {hi:.2f}" if lo is not None and hi is not None else "N/A",
        "BVPS": f"{bvps_val:.2f}" if bvps_val else "N/A"
    }
    return save_to_file(json.dumps(key_data, indent=2), save_path)



def get_company_profile(ticker: str, save_path: str, region: str = 'IN') -> str:
    """
    Retrieves company profile and description exclusively from IndianAPI.
    Returns the file path where profile is saved.
    """
    print(f"Fetching company profile for {ticker} from IndianAPI...")

    # Make API call
    content = get_cached_stock_data(ticker)
    # content = make_api_request("IndianMarket", "/stock", {"name": ticker})

    if not isinstance(content, dict):
        logging.error(f"No valid dict response for {ticker}: got {type(content)} {content}")
        content = {}

    # Unwrap if 'companyProfile' present, else use whole content
    if 'companyProfile' in content:
        content = content['companyProfile']

    # Defensive extraction of company description and mgIndustry
    company_desc = content.get('companyDescription', '')
    mg_industry = content.get('mgIndustry', '')

    # Defensive extraction of metrics from peerCompanyList
    peer_companies = content.get('peerCompanyList')
    if isinstance(peer_companies, list) and peer_companies:
        metrics = peer_companies[0]
    else:
        metrics = {}

    content_parsed = {
        'companyDescription': company_desc,
        'mgIndustry': mg_industry,
        'metrics': metrics,
    }

    save_to_file(json.dumps(content_parsed, indent=2), save_path)
    logging.info(f"Successfully saved company profile for {ticker} to {save_path}")
    return save_path


def get_competitor_analysis(ticker: str, save_path: str) -> str:
    """
    Retrieves a list of competitors (peer companies) from IndianAPI.
    Saves the list (can be empty) to the given file path.
    """
    print(f"Fetching competitors for {ticker} from IndianAPI...")

    # API call
    content = get_cached_stock_data(ticker)
    #content = make_api_request("IndianMarket", "/stock", {"name": ticker})
    if not isinstance(content, dict):
        logging.error(f"No valid dict response for {ticker}: got {type(content)} {content}")
        content = {}

    # Try to unwrap 'companyProfile'
    company_profile = content.get("companyProfile")
    if not isinstance(company_profile, dict):
        logging.warning(f"No companyProfile in response for {ticker}: {content}")
        competitors = []
    else:
        competitors = company_profile.get("peerCompanyList", [])
        if not isinstance(competitors, list):
            logging.warning(f"peerCompanyList is not a list for {ticker}: {company_profile}")
            competitors = []

    save_to_file(json.dumps(competitors, indent=2), save_path)
    logging.info(f"Successfully saved competitors for {ticker} to {save_path}")
    return save_path


def get_financial_statement(ticker: str, statement_type: str, save_path: str, fyear: int) -> str:
    """
    Retrieves a financial statement (income, balance, cash-flow) from IndianAPI.
    Always saves a file (with data or an error message).
    Returns the save_path.
    """
    print(f"Fetching {statement_type} for {ticker} from IndianAPI...")

    type_map = {
        "income_statement": "INC",
        "balance_sheet": "BAL",
        "cash_flow_statement": "CAS"
    }
    code_map = {
        "INC": "Income Statement",
        "BAL": "Balance Sheet",
        "CAS": "Cashflow Statement"
    }

    statement_code = type_map.get(statement_type)
    if not statement_code:
        msg = f"Invalid statement type: {statement_type}"
        logging.error(msg)
        save_to_file(json.dumps({"error": msg}), save_path)
        return save_path

    # Make API call
    response = get_cached_stock_data(ticker)
    # response = make_api_request("IndianMarket", "/stock", {"name": ticker})
    if not isinstance(response, dict):
        msg = f"Invalid response for {ticker}: {response}"
        logging.error(msg)
        save_to_file(json.dumps({"error": msg}), save_path)
        return save_path

    financials = response.get("financials", [])
    if not isinstance(financials, list) or not financials:
        msg = f"No financials found for {ticker}"
        logging.warning(msg)
        save_to_file(json.dumps({"error": msg}), save_path)
        return save_path


    # Parse annual financial data
    statement_data = {}
    for doc in financials:
        print(doc)
        if doc.get('Type') == 'Annual':
            fmap = doc.get("stockFinancialMap", {})
            if statement_code in fmap:
                statement_data[doc['FiscalYear']] = fmap[statement_code]
                print(statement_data)
    """if str(fyear) not in statement_data:
        msg = f"Financial data for year {fyear} not available for {ticker}"
        logging.warning(msg)
        save_to_file(json.dumps({"error": msg}), save_path)
        return save_path"""

    # Optional: merge with previous year
    def safe_dataframe(data):
        try:
            df = pd.DataFrame(data)
            if "key" in df.columns and "value" in df.columns:
                return df[["key", "value"]]
            else:
                logging.warning(f"Expected columns missing in data for {ticker} {statement_type} {fyear}: {df.columns}")
                return pd.DataFrame(columns=["key", "value"])
        except Exception as e:
            logging.error(f"Could not build DataFrame: {e}")
            return pd.DataFrame(columns=["key", "value"])

    current_year_df = safe_dataframe(statement_data[str(fyear)])
    prev_year_df = safe_dataframe(statement_data.get(str(fyear - 1), []))

    result_df = current_year_df.merge(prev_year_df, on="key", how="left", suffixes=('', '_prev'))
    result_df.columns = [code_map[statement_code], str(fyear), str(fyear - 1)]

    # Save to file
    json_data = result_df.to_dict(orient="records")
    save_to_file(json.dumps(json_data, indent=2), save_path)
    logging.info(f"Successfully saved {statement_type} for {ticker} to {save_path}")
    return save_path



def get_indian_key_metrics_raw(stock: str, save_path: str = None) -> Dict[str, Any]:
    """
    Fetch raw keyMetrics data from indianapi.in for the given stock symbol.
    Returns the data exactly as received (dictionary).
    """
    
    response = get_cached_stock_data(stock)
    #response = make_api_request("IndianMarket", "/stock", {"name": stock})
    data = response.get('keyMetrics', {})
    return save_to_file(json.dumps(data, indent=2), save_path)

def get_indian_historical_prices(ticker: str, start_date: str = None, end_date: str = None,period:str = 'max', filter: str = 'price',save_path: str = None) -> pd.DataFrame:
    """
    Fetches historical prices from the Indian API, saves JSON if path provided,
    and returns a merged DataFrame with parsed data similar to your example.
    """
    print(f"Fetching historical prices for {ticker}...")

    # Make API call
    endpoint = "/historical_data"  # adapt as needed
    params = {"stock_name": ticker, "period": period, "filter": filter}
    content = make_api_request("IndianMarket", endpoint, params)

    if save_path:
        save_to_file(json.dumps(content, indent=2), save_path)

    # Parse 'datasets' like in your second example, or adapt if structure differs
    # Let's assume Indian API returns something like: {"datasets": [{"metric": "...", "values": [...]}, ...]}
    parsed = content

    # Extract historical data
    table = None

    for metric in parsed['datasets']:
        metric_name = metric['metric']
        data = metric['values']

        # Convertir a DataFrame
        if metric_name != "Volume":
            df = pd.DataFrame(data, columns=['Date', metric_name])
        else:
            data = [[row[0], row[1], row[2]['delivery']] for row in data]
            df = pd.DataFrame(data, columns=['Date', metric_name, 'delivery'])
        df['Date'] = pd.to_datetime(df['Date']) 

        if table is None:
            table = df
        else:
            table = pd.merge(table, df, on='Date', how='left')  # Puedes usar 'inner' si prefieres solo fechas comunes

    # Ordenar y filtrar por fecha si es necesario
    table = table.sort_values('Date').reset_index(drop=True)
    if start_date is not None and end_date is not None:
        table =  table[
    (table['Date'] >= pd.to_datetime(start_date)) & 
    (table['Date'] <= pd.to_datetime(end_date))
]
    table['table'] = table['Date'].astype(str)  # Convertir a string si es necesario
    table.set_index('Date', inplace=True)  # Establecer 'Date' como índice
    table.index = table.index.astype(str) 
    return save_to_file(json.dumps(table.to_dict()), save_path) if save_path else table

def get_annual_report_sections_1(ticker: str, fyear: str, save_path: str) -> str:
    """
    Fetches sections 2 from the latest RAG for an Indian company
    and saves each section to a single text file (full path provided).
    """
    print(f"Fetching Annual Report sections for {ticker}...")

    # Make sure only the directory is created
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Fetch business overview data
    company_overview = (
        #make_api_request2("IndianMarket", "/stock", {"name": ticker})
        get_cached_stock_data(ticker)        
        .get('companyProfile', {})
        .get('companyDescription', '')
    )
    business_overview = get_annual_report_section(ticker, fyear=fyear, section="2")['text']
    full_text = f"{company_overview}\n\n{business_overview}"

    # Save to the provided file path
    save_to_file(full_text, save_path)
    logging.info(f"Successfully saved {save_path}")
    return save_path

def get_annual_report_sections_7(ticker: str, fyear: str, save_path: str) -> str:
    """
    Fetches section 1 from the latest RAG for an Indian company
    and saves it to a single text file (full file path provided in save_path).
    This is equivalent to section 7 in a US 10-K.
    """
    print(f"Fetching Annual Report sections for {ticker}...")

    # Ensure the parent directory exists, NOT the file itself
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Management Discussion and Analysis Section
    section = "1"
    management_discussion_analysis = get_annual_report_section(ticker, fyear=fyear, section=section)['text']

    save_to_file(management_discussion_analysis, save_path)
    logging.info(f"Successfully saved {save_path}")
    return save_path

def get_annual_report_sections_1a(ticker: str, fyear: str, save_path: str) -> str:
    """
    Fetches section 1A (risks) from the latest RAG for an Indian company
    and saves it to the given text file path (full path expected in save_path).
    """
    print(f"Fetching Annual Report sections for {ticker}...")

    # Ensure the parent directory of the file exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Risk Section
    section = "1"
    risks = "\n".join(
        perform_similarity_search(
            ticker, fyear=fyear, section=section, question="What are the company's risks?"
        )['chunks']
    )
    save_to_file(risks, save_path)
    logging.info(f"Successfully saved {save_path}")
    return save_path


def generate_share_performance_chart_indian_market(ticker: str, fyear: str, save_path: str) -> str:
    return get_indian_share_performance(ticker, f"{fyear}-12-31", save_path)

def generate_pe_eps_chart_indian_market(ticker: str, fyear: str, save_path: str) -> str:
    return get_pe_eps_performance_indian_market(ticker, f"{fyear}-12-31", save_path)

def _get_financial_statement_for_table(
    ticker: str,
    statement_type: str,
    start_year: int = None,
    years: int = 1,
) -> pd.DataFrame:
    """
    Retrieves financial statements (income, balance, cash-flow) for multiple years from IndianAPI.
    Args:
        ticker: Stock ticker symbol.
        statement_type: One of "income_statement", "balance_sheet", "cash_flow_statement".
        save_path: Path to save the JSON output. If None, no file saved.
        start_year: Starting fiscal year (int).
        years: Number of years to retrieve, counting backwards from start_year.
    Returns:
        Pandas DataFrame with 'key' as index and columns for each year.
    """
    print(f"Fetching {statement_type} for {ticker} from IndianAPI from year {start_year} for {years} years...")

    # Map statement_type to IndianAPI code
    type_map = {
        "income_statement": "INC",
        "balance_sheet": "BAL",
        "cash_flow_statement": "CAS"
    }
    statement_code = type_map.get(statement_type)
    if not statement_code:
        raise ValueError(f"Invalid statement type: {statement_type}")

    # Fetch data
    response = get_cached_stock_data(ticker)
    #response = make_api_request2("IndianMarket", "/stock", {"name": ticker})

    financials = response.get("financials", [])

    # Filter only annual data that contains the required statement
    statement_data = {
        int(doc['FiscalYear']): doc['stockFinancialMap'][statement_code]
        for doc in financials
        if doc['Type'] == 'Annual' and statement_code in doc.get("stockFinancialMap", {})
    }

    if start_year is None:
        # If no start_year given, pick the most recent year available
        start_year = max(statement_data.keys()) if statement_data else None

    if start_year not in statement_data:
        raise ValueError(f"Financial data for year {start_year} not available for {ticker}")

    # Collect years to fetch (descending order)
    years_to_fetch = [start_year - i for i in range(years)]

    # Build dict of DataFrames by year
    dfs = {}
    for yr in years_to_fetch:
        if yr in statement_data:
            df_year = pd.DataFrame(statement_data[yr])[['key', 'value']].set_index('key')
            dfs[yr] = df_year.rename(columns={'value': str(yr)})
        else:
            print(f"Warning: Data for year {yr} not found for {ticker}")

    if not dfs:
        raise ValueError(f"No financial data found for years requested for {ticker}")

    # Merge all years on 'key' index
    result_df = pd.concat(dfs.values(), axis=1)
    for column in result_df.columns:
        result_df[column] = pd.to_numeric(result_df[column], errors='coerce')

    return result_df

def get_financial_metrics_indian_market(ticker: str, fyear: int, save_path: str, years: int = 5) -> str:
    """
    Calculates and saves financial metrics for a company from IndianAPI data.
    Always writes a file (with metrics or with an error message).
    Returns the path to the saved file.
    """
    try:
        print(fyear-years)
        df = _get_financial_statement_for_table(
            ticker=ticker, 
            statement_type="income_statement", 
            start_year=fyear, 
            years=years
        )


        if not isinstance(df, pd.DataFrame) or df.empty:
            msg = f"No income statement data found for {ticker}."
            logging.warning(msg)
            save_to_file(json.dumps({"error": msg}), save_path)
            return save_path

        # Flatten columns if MultiIndex or tuple
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(-1)
        elif any(isinstance(col, tuple) for col in df.columns):
            df.columns = [col[-1] if isinstance(col, tuple) else col for col in df.columns]
        df.columns = df.columns.astype(str)

        year_cols = [col for col in df.columns if str(col).isdigit()]
        metrics = {}

        def safe_loc(row_name, default=0):
            return df.loc[row_name] if row_name in df.index else pd.Series([default]*len(df.columns), index=df.columns)

        revenue = safe_loc('Revenue')
        net_income = safe_loc('NetIncome')
        interest_expense = safe_loc('InterestExp(Inc)Net-OperatingTotal')
        depreciation = safe_loc('Depreciation/Amortization')
        ebit = safe_loc('OperatingIncome')
        taxes = safe_loc('ProvisionforIncomeTaxes')
        shares = safe_loc('DilutedWeightedAverageShares')
        eps = safe_loc('DilutedEPSExcludingExtraOrdItems')

        metrics['Revenue'] = round(revenue, 2)
        metrics['Net Income'] = round(net_income, 2)
        metrics['EBIT'] = round(ebit, 2)
        with pd.option_context("mode.use_inf_as_na", True):
            metrics['EBIT Margin'] = round((ebit / revenue) * 100, 2)
            metrics['Net Income Margin'] = round((net_income / revenue) * 100, 2)
            metrics['Effective Tax Rate'] = round((taxes / (net_income + taxes)) * 100, 2)
            metrics['Interest Coverage'] = round((ebit / interest_expense), 2)
        metrics['EBITDA'] = round(ebit + depreciation, 2)
        metrics['EPS'] = round(eps, 2)
        metrics['Shares'] = round(shares, 2)

        metrics_df = pd.DataFrame(metrics)
        # Only use columns for available years, and transpose for structure
        metrics_df = metrics_df.loc[year_cols].T if year_cols else metrics_df.T

        df_json = metrics_df.to_dict(orient="index")
        save_to_file(json.dumps({
            "currency": 'crore',
            "company_name": ticker.upper(),
            "metrics": df_json
        }, indent=2), save_path)
        logging.info(f"Financial metrics saved to {save_path}")
    except Exception as e:
        msg = f"Failed to calculate/save financial metrics for {ticker}: {e}"
        logging.error(msg)
        save_to_file(json.dumps({"error": msg}), save_path)

    return save_path

