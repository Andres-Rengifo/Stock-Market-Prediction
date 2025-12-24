import pandas as pd

from data_pipeline import load_data

df_metadata, historical_data = load_data()
# Copy all historical dataframes at once
dfs = {symbol: df.copy() for symbol, df in historical_data.items()}

def clean_data(df):
    """Clean the historical data by handling missing values."""

    df.index = pd.to_datetime(df.index)  # Ensure index is datetime

    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Close'])  # Drop rows where 'Close' is NaN

    df = df.fillna(method='ffill')    # Forward fill to handle other NaNs

    return df

def engineering_features(df):
    df['Daily_Return'] = df['Close'].pct_change()                      # Daily returns
    df['SMA20'] = df['Close'].rolling(window=20).mean()               # 20-day Simple Moving Average
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()       # 12-day Exponential Moving Average
    df['Volatility20'] = df['Daily_Return'].rolling(window=20).std()  # 20-day rolling volatility
    return df

def process_data(dfs, clean_data, engineering_features):
    print("Cleaning historical data...")
    for symbol in dfs:
        df = dfs[symbol]
        df = clean_data(df)
        df = engineering_features(df)
        dfs[symbol] = df
    print("Cleaning and feature engineering complete.")
    return dfs

def summary_statistics(dfs, symbol):
    series = dfs[symbol]["Daily_Return"]

    stats = pd.Series({
        "mean": series.mean(),
        "std": series.std(),
        "min": series.min(),
        "max": series.max(),
        "skew": series.skew()
    })

    return stats

dfs = process_data(dfs, clean_data, engineering_features)




#plt.title("Correlation Matrix of Daily Returns")
#plt.show()

#print("Summary statistics for FTSE 100 Index:")
#print(summary_statistics(dfs, "^FTSE"))

#std_times_two = dfs["TSLA"]["Daily_Return"].std() * 2
#print(std_times_two)