# tools/charting.py
import os
import mplfinance as mpf
import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
from typing import Annotated, List, Tuple
from pandas import DateOffset
from datetime import datetime, timedelta
import logging

# Import our standardized API toolkit
from .global_API_toolkit import make_api_request, make_api_request2

# --- Helper function to process historical data ---
def _get_historical_data_df(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetches historical data using the correct stable endpoint from the toolkit."""
    
    # --- MODIFICATION: Use the correct endpoint you provided ---
    hist_data_json = make_api_request(
        "FMP",
        "/historical-price-eod/full", # This is the correct endpoint path
        {"symbol": ticker, "from": start_date, "to": end_date}
    )
    
    # The rest of the function remains the same
    # It will now receive the correct data to process.
    if not hist_data_json or not isinstance(hist_data_json, list) or not hist_data_json:
        logging.warning(f"Could not retrieve historical data for {ticker}")
        return pd.DataFrame()
    
    # The response for this endpoint is a list directly, not nested in a dict
    df = pd.DataFrame(hist_data_json)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        return df
        
    return pd.DataFrame()


def _get_historical_prices_df_indian_market(ticker: str, start_date: str, end_date: str, period:str = 'max', filter:str = 'price') -> pd.DataFrame:
    """
    Fetches historical prices from the Indian API, saves JSON if path provided,
    and returns a merged DataFrame with parsed data similar to your example.
    """
    print(f"Fetching historical prices for {ticker}...")

    # Make API call
    endpoint = "/historical_data"  # adapt as needed
    params = {"stock_name": ticker, "period": period, "filter": filter}
    content = make_api_request2("IndianMarket", endpoint, params)


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
    return table

# --- Charting Functions ---

def plot_stock_price_chart(
    ticker_symbol: Annotated[str, "Ticker symbol"],
    start_date: Annotated[str, "Start date in 'YYYY-MM-DD' format"],
    end_date: Annotated[str, "End date in 'YYYY-MM-DD' format"],
    save_path: Annotated[str, "File path for saving the plot"],
    **kwargs # To accept other optional plotting args
) -> str:
    """
    Plots a stock price chart using mplfinance and data from the global_api_toolkit.
    """
    stock_data = _get_historical_data_df(ticker_symbol, start_date, end_date)
    
    if stock_data.empty:
        error_msg = f"Error: No historical stock data for {ticker_symbol} to plot."
        logging.error(error_msg)
        return error_msg

    # Ensure columns are named correctly for mplfinance
    stock_data.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)

    plot_params = {
        "type": kwargs.get('type', 'candle'),
        "style": kwargs.get('style', 'yahoo'),
        "title": f"{ticker_symbol} Stock Price",
        "ylabel": "Price",
        "volume": True,
        "ylabel_lower": "Volume",
        "mav": kwargs.get('mav'),
        "savefig": save_path,
    }
    
    filtered_params = {k: v for k, v in plot_params.items() if v is not None}
    mpf.plot(stock_data, **filtered_params)
    
    return f"Stock price chart saved to <img src='{save_path}'>"


# This function is in your tools/charting.py file

def get_share_performance(
    ticker_symbol: Annotated[str, "Ticker symbol"],
    filing_date: Annotated[str | datetime, "Filing date in 'YYYY-MM-DD' format"],
    save_path: Annotated[str, "File path for saving the plot"],
    benchmark_ticker: Annotated[str, "Benchmark index ticker, e.g., SPY"] = "SPY"
) -> str:
    """
    Plots the stock performance against a benchmark index over the past year,
    ensuring data is aligned for accurate comparison.
    """
    if isinstance(filing_date, str):
        filing_date = datetime.strptime(filing_date, "%Y-%m-%d")

    start = (filing_date - timedelta(days=365)).strftime("%Y-%m-%d")
    end = filing_date.strftime("%Y-%m-%d")

    # Fetch data using the helper function
    target_df = _get_historical_data_df(ticker_symbol, start, end)
    benchmark_df = _get_historical_data_df(benchmark_ticker, start, end)
    
    if target_df.empty or benchmark_df.empty:
        error_msg = f"Could not retrieve performance data for {ticker_symbol} or benchmark {benchmark_ticker}."
        logging.error(error_msg)
        return error_msg

    # --- MODIFICATION START: Align and process data robustly ---

    # Combine both 'close' price series into a single DataFrame
    combined_df = pd.DataFrame({
        'target': target_df['close'],
        'benchmark': benchmark_df['close']
    })

    # Forward-fill any missing values that might occur on non-overlapping days
    combined_df.ffill(inplace=True)
    combined_df.dropna(inplace=True) # Drop any remaining NaN if they exist at the start

    if combined_df.empty:
        error_msg = f"Data for {ticker_symbol} and {benchmark_ticker} could not be aligned."
        logging.error(error_msg)
        return error_msg
        
    # Normalize the data from the first valid data point and calculate percentage change
    normalized_df = (combined_df / combined_df.iloc[0]) * 100

    # --- MODIFICATION END ---

    # Get company name from profile
    profile_response = make_api_request("FMP", "/profile", {"symbol": ticker_symbol})
    info = profile_response[0] if profile_response and isinstance(profile_response, list) else {}
    company_name = info.get("companyName", ticker_symbol)
    
    # Plotting logic now uses the aligned and normalized data
    plt.figure(figsize=(14, 7))
    plt.plot(normalized_df.index, normalized_df['target'], label=f'{company_name} Indexed Performance', color="blue")
    plt.plot(normalized_df.index, normalized_df['benchmark'], label=f"{benchmark_ticker} Indexed Performance", color="red")
    
    plt.title(f'{company_name} vs {benchmark_ticker} - Indexed Performance Over the Past Year')
    plt.xlabel("Date")
    plt.ylabel("Normalized Price (Base 100)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return f"Share performance chart saved to <img src='{save_path}'>"

def get_indian_share_performance(
    ticker_symbol: str,
    filing_date: str | datetime,
    save_path: str,
    benchmark_ticker: str = "NIFTYBEES.NS"
) -> str:
    """
    Plots the indexed performance of an Indian stock against a benchmark ETF (default: NIFTYBEES.NS)
    over the past year, rebased to 100 for comparison.
    """
    # Convert date string to datetime
    if isinstance(filing_date, str):
        filing_date = datetime.strptime(filing_date, "%Y-%m-%d")

    start = filing_date - timedelta(days=365)
    end = filing_date

    # Append .NS for NSE if not already present
    if not ticker_symbol.endswith(".NS"):
        ticker_symbol += ".NS"

    # Fetch data
    try:
        # Target stock data
        target_df = yf.download(ticker_symbol, start=start, end=end)
        target_df.reset_index(inplace=True, drop=False)
        target_df.columns = target_df.columns.get_level_values(0)
        target_df = target_df[['Date', 'Close']]
        target_df.set_index('Date', inplace=True)

        # Benchmark data
        benchmark_df = yf.download(benchmark_ticker, start=start, end=end)
        benchmark_df.reset_index(inplace=True, drop=False)
        benchmark_df.columns = benchmark_df.columns.get_level_values(0)
        benchmark_df = benchmark_df[['Date', 'Close']]
        benchmark_df.set_index('Date', inplace=True)

    except Exception as e:
        logging.error(f"Data download failed: {e}")
        return f"Data download failed: {e}"

    if target_df.empty or benchmark_df.empty:
        error_msg = f"Could not retrieve data for {ticker_symbol} or {benchmark_ticker}."
        logging.error(error_msg)
        return error_msg

    # Combine and align
    combined_df = pd.DataFrame({
        'target': target_df['Close'],
        'benchmark': benchmark_df['Close']
    })
    combined_df.ffill(inplace=True)
    combined_df.dropna(inplace=True)

    if combined_df.empty:
        error_msg = f"Aligned data is empty after processing."
        logging.error(error_msg)
        return error_msg

    # Normalize to 100
    normalized_df = (combined_df / combined_df.iloc[0]) * 100

    # Plot
    plt.figure(figsize=(14, 7))
    plt.plot(normalized_df.index, normalized_df['target'], label=f'{ticker_symbol.replace('.NS', '')} Indexed Performance', color="blue")
    plt.plot(normalized_df.index, normalized_df['benchmark'], label=f'{benchmark_ticker.replace('.NS', '')} Indexed Performance', color="red")

    plt.title(f'{ticker_symbol.replace('.NS', '')} vs {benchmark_ticker.replace('.NS', '')} - Indexed Performance Over the Past Year')
    plt.xlabel("Date")
    plt.ylabel("Normalized Price (Base 100)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return f"Share performance chart saved to <img src='{save_path}'>"

# This function goes in your tools/charting.py file

def get_pe_eps_performance(
    ticker_symbol: Annotated[str, "Ticker symbol"],
    filing_date: Annotated[str | datetime, "Filing date in 'YYYY-MM-DD' format"],
    save_path: Annotated[str, "File path for saving the plot"],
    years: int = 4
) -> str:
    """Plots the PE ratio and EPS performance over the past n years."""

    

    # 1. Fetch income statements to get EPS data from the API
    income_response = make_api_request("FMP", "/income-statement", {"symbol": ticker_symbol, "period": "annual", "limit": years + 1})
    
    # Defensively check the response
    if not income_response or not isinstance(income_response, list):
        error_msg = f"Could not retrieve income statements for {ticker_symbol}"
        logging.warning(error_msg)
        return error_msg
    
    income_statements = pd.DataFrame(income_response)
    if 'date' not in income_statements.columns:
        error_msg = f"Malformed income statement data for {ticker_symbol}, 'date' column missing."
        logging.warning(error_msg)
        return error_msg
        
    # 2. Prepare the EPS data and convert its index to proper datetime objects
    eps = income_statements.set_index('date')['eps'].sort_index()
    eps.index = pd.to_datetime(eps.index) # This ensures the index is datetime, not string

    # 3. Fetch historical prices to calculate the PE ratio
    if isinstance(filing_date, str):
        filing_date = datetime.strptime(filing_date, "%Y-%m-%d")
        
    start = (filing_date - timedelta(days=years*365)).strftime("%Y-%m-%d")
    end = filing_date.strftime("%Y-%m-%d")
    
    price_df = _get_historical_data_df(ticker_symbol, start, end)
    if price_df.empty:
        error_msg = f"Could not retrieve historical prices for {ticker_symbol}"
        logging.warning(error_msg)
        return error_msg
    
    # 4. Align stock prices with EPS report dates
    pe_ratios = {}
    # The loop now correctly iterates over 'report_date' which is already a datetime object
    for report_date, eps_val in eps.items():
        # The unnecessary and error-causing strptime() line has been removed.
        try:
            # Find the stock price on or immediately after the EPS report date
            price_at_date = price_df.loc[price_df.index >= report_date, 'close'].iloc[0]
            if eps_val and eps_val != 0:
                pe_ratios[report_date] = price_at_date / eps_val
        except IndexError:
            # This handles cases where there's no stock data after the last EPS date
            continue
            
    if not pe_ratios:
        error_msg = f"Could not calculate P/E ratios for {ticker_symbol}. Check data alignment."
        logging.warning(error_msg)
        return error_msg

    pe_series = pd.Series(pe_ratios).sort_index()


    # 5. Plotting Logic
    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax1.plot(pe_series.index, pe_series.values, color="blue", marker='o', label="P/E Ratio")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("P/E Ratio", color="blue")
    
    ax2 = ax1.twinx()
    ax2.plot(eps.index, eps.values, color="red", marker='x', linestyle='--', label="EPS")
    ax2.set_ylabel("EPS ($)", color="red")
    
    # Get company name for the title
    profile_response = make_api_request("FMP", "/profile", {"symbol": ticker_symbol})
    company_name = profile_response[0].get("companyName", ticker_symbol) if profile_response and isinstance(profile_response, list) else ticker_symbol
    
    plt.title(f"{company_name} P/E Ratio and EPS Performance")
    fig.tight_layout()
    
    # 6. Save the figure to a file and close it to prevent display
    plt.savefig(save_path)
    plt.close(fig) 
    
    return f"P/E and EPS performance chart saved to <img src='{save_path}'>"

def get_pe_eps_performance_indian_market(
    ticker_symbol: Annotated[str, "Ticker symbol"],
    filing_date: Annotated[str | datetime, "Filing date in 'YYYY-MM-DD' format"],
    save_path: Annotated[str, "File path for saving the plot"],
    years: int = 4
) -> str:
    """Plots the PE ratio and EPS performance over the past n years."""

    year = int(filing_date.strftime("%Y")) if isinstance(filing_date, datetime) else int(filing_date.split("-")[0])

    # 1. Fetch income statements to get EPS data from the API

    # Make the API call
    response = make_api_request2("IndianMarket", "/stock", {"name": ticker_symbol})
    financials = response.get("financials", [])

    # Parse annual financial data
    statement_data = {
        doc['EndDate']: doc['stockFinancialMap']['INC']
        for doc in financials
        if doc['Type'] == 'Annual' and 'INC' in doc.get("stockFinancialMap", {})
    }

    df = pd.DataFrame()
    for date, income_statement in statement_data.items():
        if int(date[:4]) <= year:
            if df.empty:
                df = pd.DataFrame(income_statement)[['key', 'value']].rename(columns={'key': 'metric', 'value': date})
            else:
                df = df.merge(
                    pd.DataFrame(income_statement)[['key', 'value']].rename(columns={'key': 'metric', 'value': date}),
                    on='metric',
                    how='outer'
                )

    df.set_index('metric', inplace=True)
    df = df.T
    eps = df['DilutedEPSExcludingExtraOrdItems'].sort_index(ascending=True).rename('EPS')
    eps = eps.astype(float)
    
    pe_df = pd.DataFrame(make_api_request2("IndianMarket", "/historical_data", {'stock_name': ticker_symbol, 'period': 'max', 'filter': 'pe'})['datasets'][1]['values'], columns = ['Date', 'PE'])
    pe_df['Date'] = pd.to_datetime(pe_df['Date'])
    list_dates = list(eps.index)

    pes = {}
    
    for date in list_dates:
        d = pd.to_datetime(date)
        pe_df_upper = pe_df[(pe_df['Date'] >= d)]
        pe_df_upper = pe_df_upper.sort_values(by='Date', ascending=True).reset_index(drop=True)

        pe_df_lower = pe_df[(pe_df['Date'] <= d)]
        pe_df_lower = pe_df_lower.sort_values(by='Date', ascending=True).reset_index(drop=True)

        upper = pe_df_upper.head(1)
        lower = pe_df_lower.tail(1)

        pe = pd.concat([lower, upper], ignore_index=True)
        pe['PE'] = pe['PE'].astype(float)

        pe_list = pe['PE'].to_list() 
        pe_dates = pe['Date'].to_list()

        d2_d1 = (pe_dates[1] - pe_dates[0]).days
        d_d1 = (d - pe_dates[0]).days

        if d_d1 <= 0 or d2_d1 <= 0:
            pe_in_d=pe_list[0]
        else:
            pe_in_d=round(pe_list[0] + (pe_list[1] - pe_list[0]) * (d_d1 / d2_d1), 2)

        pes[date] = pe_in_d
    
    pe_series = pd.Series(pes).sort_index(ascending=True)
    pe_series.index = pd.to_datetime(pe_series.index)
    filing_date = pd.to_datetime(filing_date)

    pe_series = pe_series[pe_series.index <= filing_date] 
    pe_series = pe_series.tail(years)  # Keep only the last n years of data

    eps.index = pd.to_datetime(eps.index)
    eps = eps[eps.index <= filing_date]
    eps = eps.tail(years)  # Filter EPS to the last n years

    print(f"PE Series: {pe_series}"
          )
    print(f"EPS Series: {eps}"
          ) 
    # 5. Plotting Logic
    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax1.plot(pe_series.index, pe_series.values, color="blue", marker='o', label="P/E Ratio")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("P/E Ratio", color="blue")

    ax2 = ax1.twinx()
    ax2.plot(eps.index, eps.values, color="red", marker='x', linestyle='--', label="EPS")
    ax2.set_ylabel("EPS ($)", color="red")

    plt.title(f"{ticker_symbol} P/E Ratio and EPS Performance")
    fig.tight_layout()

    plt.savefig(save_path)
    plt.close(fig)

    return f"P/E and EPS performance chart saved to <img src='{save_path}'>"