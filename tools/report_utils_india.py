import json
import re
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, Any, Annotated, Tuple
from collections import defaultdict
import os
import logging


# Import the core API calling functions from your toolkit
from .global_API_toolkit import make_api_request2, call_sec_utility, save_to_file
from .charting import *
from .rag_api_utils import get_annual_report_section, perform_similarity_search


def get_key_data_india(
    stock_name: str,
    save_path: str
) -> str:
    from datetime import datetime, timedelta
    import pandas as pd, json

    # Fecha
    r = make_api_request2("IndianMarket", "/stock", {"name": stock_name})

    if not r:
        return save_to_file(json.dumps({"error": "No data"}), save_path)

    profile = r.get("companyProfile", {})
    curr = 'CROE'
    
    # Precios
    curp = r.get("currentPrice", {})
    price = curp.get("NSE") or curp.get("BSE") or 0
    fifty_two_week = [data  for data in r.get('keyMetrics').get('priceandVolume') if data.get('key') == '52WeekLow' or data.get('key') == '52WeekHigh']
    lo, hi = [float(data.get('value')) for data in fifty_two_week]

    # Ratios/BVPS
    km = r.get("keyMetrics", {})
    bvps = km.get("bookValuePerShare", 0)

    # Market Cap probablemente en financials
    mc = r.get('companyProfile').get('peerCompanyList')[0].get('marketCap')

    # Analista
    av = r.get("analystView", {})
   

    key_data = {
      "Rating": av,
      "Currency": curr,
      "Closing Price": price,
      "Market Cap (Millions)": f"{mc/1e6:.2f}",
      "52 Week Price Range": f"{lo:.2f} - {hi:.2f}",
      "BVPS": f"{bvps:.2f}"
    }
    return save_to_file(json.dumps(key_data, indent=2), save_path)


def get_company_profile(ticker: str, save_path: str, region: str = 'IN', **kwargs) -> str:
    """
    Retrieves company profile and description exclusively from IndianAPI.
    """
    print(f"Fetching company profile for {ticker} from IndianAPI...")

    # IndianAPI uses "name" instead of "symbol"
    content = make_api_request2("IndianMarket", "/stock", {"name": ticker})
    content = content.get('companyProfile', content) if isinstance(content, dict) else content
    content_parsed = {'companyDescription': content.get('companyDescription', ''),
                  'mgIndustry': content.get('mgIndustry', ''), 'metrics': content.get('peerCompanyList', {})[0]}

    return save_to_file(json.dumps(content_parsed, indent=2), save_path)

def get_competitor_analysis(ticker: str, save_path: str, **kwargs) -> str:
    """
    Retrieves a list of competitors (peer companies) from IndianAPI.
    """
    print(f"Fetching competitors for {ticker} from IndianAPI...")

    # Llamada a la API
    content = make_api_request2("IndianMarket", "/stock", {"name": ticker})

    # Extraer lista de competidores
    competitors = content.get("companyProfile", {}).get("peerCompanyList", [])

    return save_to_file(json.dumps(competitors, indent=2), save_path)

def get_financial_statement(ticker: str, statement_type: str, save_path:str ,year: int, **kwargs) -> str:
    """
    Retrieves a financial statement (income, balance, cash-flow) from IndianAPI.
    """
    print(f"Fetching {statement_type} for {ticker} from IndianAPI...")

    # Map from input to IndianAPI codes
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
        raise ValueError(f"Invalid statement type: {statement_type}")

    # Make the API call
    response = make_api_request2("IndianMarket", "/stock", {"name": ticker})
    financials = response.get("financials", [])

    # Parse annual financial data
    statement_data = {
        doc['FiscalYear']: doc['stockFinancialMap'][statement_code]
        for doc in financials
        if doc['Type'] == 'Annual' and statement_code in doc.get("stockFinancialMap", {})
    }

    if str(year) not in statement_data:
        raise ValueError(f"Financial data for year {year} not available for {ticker}")

    # Optional: merge with previous year
    current_year_df = pd.DataFrame(statement_data[str(year)])[['key', 'value']]
    previous_df = pd.DataFrame(statement_data.get(str(year - 1), []))[['key', 'value']] if str(year - 1) in statement_data else pd.DataFrame(columns=["key", "value"])

    result_df = current_year_df.merge(previous_df, on="key", how="left")
    result_df.columns = [code_map[statement_code], str(year), str(year - 1)]

    # Save to file
    json_data = result_df.to_dict(orient="records")
    return save_to_file(json.dumps(json_data, indent=2), save_path)



def get_indian_key_metrics_raw(stock: str, save_path: str = None) -> Dict[str, Any]:
    """
    Fetch raw keyMetrics data from indianapi.in for the given stock symbol.
    Returns the data exactly as received (dictionary).
    """
    
    response = make_api_request2("IndianMarket", "/stock", {"name": stock})
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
    content = make_api_request2("IndianMarket", endpoint, params)

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

def get_annual_report_sections(ticker: str, fyear: str, save_path: str, **kwargs) -> str:
    """
    Fetches sections 1, 1A, and 7 from the latest 10-K filing for a US company
    and saves each section to a separate text file.
    """
    print(f"Fetching Annual Report sections for {ticker}...")
    
    os.makedirs(save_path, exist_ok=True)

    # Management Discussion and Analysis Section
    section = "1"
    management_discussion_analysis = get_annual_report_section(ticker, fyear=fyear, section=section)['text'] 
    file_name = f"management_and_discussion_analysis.txt"
    full_path = os.path.join(save_path, file_name)
    save_to_file(management_discussion_analysis, full_path)
    logging.info(f"Successfully saved {full_path}")

    # Risk Section
    section = "1"
    risks = "\n".join(perform_similarity_search(ticker, fyear=fyear, section=section, question="What are the company's risks?")['chunks'])        
    file_name = f"company_risks.txt"
    full_path = os.path.join(save_path, file_name)
    save_to_file(risks, full_path)
    logging.info(f"Successfully saved {full_path}")

    # Business Overview Section
    section = "2"
    company_oveview = make_api_request2("IndianMarket", "/stock", {"name": ticker}).get('companyProfile').get('companyDescription')
    business_overview = get_annual_report_section(ticker, fyear=fyear, section=section)['text']
    business_overview = f"{company_oveview}\n\n{business_overview}"
    file_name = f"business_overview.txt"
    full_path = os.path.join(save_path, file_name)
    save_to_file(business_overview, full_path)
    logging.info(f"Successfully saved {full_path}")

def generate_share_performance_chart_indian_market(ticker: str, fyear: str, save_path: str, **kwargs) -> str:
    return get_indian_share_performance(ticker, f"{fyear}-12-31", save_path)

def generate_pe_eps_chart_indian_market(ticker: str, fyear: str, save_path: str, **kwargs) -> str:
    return get_pe_eps_performance_indian_market(ticker, f"{fyear}-12-31", save_path)
