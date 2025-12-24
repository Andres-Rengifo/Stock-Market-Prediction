# Bulding the machine learning model

from eda import dfs
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np

# Create target variable for prediction
# the target is the percentage change in closing price the next day
def create_target(df):
    df = df.copy()

    # df['close'].shift(-1) shifts the 'close' column up by one row
    # so that we are comparing today's close with tomorrow's close
    df['Target'] = df['Close'].shift(-1) / df['Close'] - 1

    # Remove rows with NaN target values (the last row will have NaN target)
    df = df.dropna(subset=['Target'])

    return df

def prepare_features(df, feature_cols=None):
    df = df.copy()

    # Select relevant features for the model
    if feature_cols is None:
        # Daily_Return -> recent price change, SMA20 -> short-term trend, EMA12 -> recent momentum, Volatility20 -> risk measure
        feature_cols = ['Daily_Return', 'SMA20', 'EMA12', 'Volatility20']
    # Separate features and target variable
    X = df[feature_cols]
    y = df['Target']

    return X, y

def train_test_split_time_series(X, y, test_size=0.2):

    # Determine the split index
    split_index = int(len(X) * (1 - test_size))

    # Split the data into training and testing sets
    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):

    # Initialize the Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Fit the model to the training data
    model.fit(X_train, y_train)

    return model

def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("Regression Metrics:")
    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R² Score:", r2)

    return y_pred

def plot_predictions(y_test, y_pred, symbol):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test, label='Actual', color='blue')
    plt.plot(y_test.index, y_pred, label='Predicted', color='red', alpha=0.7)
    plt.title(f"Actual vs Predicted Daily Returns for {symbol}")
    plt.xlabel("Date")
    plt.ylabel("Daily Return")
    plt.legend()
    plt.show()

def run_pipeline(dfs, symbol):
    print(f"Running pipeline for {symbol}...")
    df = dfs[symbol]
    df = create_target(df)
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split_time_series(X, y)
    model = train_model(X_train, y_train)
    y_pred = evaluate_model(model, X_test, y_test)
    plot_predictions(y_test, y_pred, symbol)
    print(f"Pipeline complete for {symbol}.")

run_pipeline(dfs, 'LLOY.L')
run_pipeline(dfs, 'TSLA')
run_pipeline(dfs, 'SHEL.L')
run_pipeline(dfs, 'RRL.XC')
run_pipeline(dfs, 'TSCO.L')
run_pipeline(dfs, 'JD.L')
