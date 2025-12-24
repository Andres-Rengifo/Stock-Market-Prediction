import yfinance as yf
from datetime import datetime
from pytz import timezone
import pandas as pd
import numpy as np

# Parameters
START_DATE = "2024-01-01"
END_DATE = "2024-06-01"

# Define ticker objects and their corresponding company names
tickers = {
    "TSLA": "Tesla",
    "LLOY.L": "Lloyds Banking Group",
    "SHEL.L": "Shell",
    "RRL.XC": "Rolls Royce",
    "TSCO.L": "Tesco",
    "JD.L": "JD Sports",
    "VODL.XC": "Vodafone",
    "BARC.L": "Barclays",
    "GSK.L": "GSK",
    "HSBA.L": "HSBC",
    "AZN.L": "AstraZeneca",
    "IMB.L": "Imperial Brands",
    "^FTSE": "FTSE 100 Index"
}

def fetch_metadata(ticker_obj, name):
    """Fetch metadata for a given ticker object."""
    info = ticker_obj.info
    mkt_time = datetime.fromtimestamp(info.get('regularMarketTime'), timezone(info['exchangeTimezoneName']))
    
    return {
        "symbol": ticker_obj.ticker,
        "name": name,
        "sector": info.get('sector', 'N/A'),
        "industry": info.get('industry', 'N/A'),
        "market": info.get('market', 'N/A'),
        "country": info.get('country', 'N/A'),
        "market_time": mkt_time,
        "previous_close": info.get('previousClose', np.nan),
        "open": info.get('open', np.nan),
        "day_low": info.get('dayLow', np.nan),
        "day_high": info.get('dayHigh', np.nan),
        "volume": info.get('volume', np.nan),
        "market_cap": info.get('marketCap', np.nan),
        "beta": info.get('beta', np.nan),
        "pe_ratio": info.get('trailingPE', np.nan),
        "dividend_yield": info.get('dividendYield', np.nan)
    }

# Fetch historical data for a given ticker object. (ticker -> object representing a given stock)
# with start and end dates (defined above)
def fetch_history(ticker_obj, start=START_DATE, end=END_DATE):
    """Fetch historical data for a given ticker object."""

    # .history method from yfinance.
    return ticker_obj.history(start=start, end=end, interval="1d")

def load_data():
    
    # Fetch live metadata and historical data
    # company_data will store metadata for all companies
    company_data = []
    # historical_data will store historical data for all companies
    historical_data = {}

    # Loop through each ticker to fetch data
    for symbol, name in tickers.items():
        # Creates a ticker for that stock
        ticker_obj = yf.Ticker(symbol)
        metadata = fetch_metadata(ticker_obj, name)
        company_data.append(metadata)
        historical_data[symbol] = fetch_history(ticker_obj)

    # Create DataFrame from metadata for the live data
    df_metadata = pd.DataFrame(company_data)
    return df_metadata, historical_data

