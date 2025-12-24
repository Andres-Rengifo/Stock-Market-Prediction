import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import seaborn as sns

from data_pipeline import load_data
from eda import process_data
from eda import clean_data
from eda import engineering_features

df_metadata, historical_data = load_data()
dfs = {symbol: df.copy() for symbol, df in historical_data.items()}
dfs = process_data(dfs, clean_data, engineering_features)
returns_df = pd.DataFrame({symbol: dfs[symbol]["Daily_Return"] for symbol in dfs})

def plot_correlation_matrix(returns_df):
    corr_matrix = returns_df.corr()
    print(corr_matrix)

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        corr_matrix, 
        annot=True, 
        fmt=".2f", 
        cmap='coolwarm', 
        cbar_kws={"label": "correlation coefficient"},
        linewidths=0,
        square=True
    )

    plt.title("Correlation Matrix of Daily Returns")
    plt.show()

    return corr_matrix

plot_correlation_matrix(returns_df) 

def plot_graph(dfs, symbol):
    sma20_plt = mpf.make_addplot(dfs[symbol]["SMA20"], color='orange')
    ema12_plt = mpf.make_addplot(dfs[symbol]["EMA12"], color='blue')
    vol20_plt = mpf.make_addplot(dfs[symbol]["Volatility20"], panel=1, color='purple', ylabel='Volatility')
    daily_ret_plt = mpf.make_addplot(dfs[symbol]["Daily_Return"], panel=2, color='green', ylabel='Daily Return')

    mpf.plot(
        dfs[symbol],
        type="candle",
        style="yahoo",
        title=f"{symbol} Candlestick Chart (Cleaned Data)",
        addplot=[sma20_plt, ema12_plt, vol20_plt, daily_ret_plt]
    )
    

plot_graph(dfs, "LLOY.L")


