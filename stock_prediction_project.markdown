# Stock Market Time Series Prediction Project

## Project Overview
This project implements a time series prediction model for stock prices using Long Short-Term Memory (LSTM) neural networks. It includes data preprocessing, model training, prediction, and visualization of results.

## Dependencies
- Python 3.8+
- pandas
- numpy
- yfinance
- scikit-learn
- tensorflow
- matplotlib
- datetime

## Installation
1. Install required packages:
```bash
pip install pandas numpy yfinance scikit-learn tensorflow matplotlib
```

## Project Structure
- `stock_prediction.py`: Main script containing the prediction model
- `requirements.txt`: Dependencies list
- `README.md`: Project documentation

## Code Implementation
Below is the main Python script that implements the stock market prediction:

<xaiArtifact artifact_id="d21a31e2-147d-4a39-9fc3-9e079f79542c" artifact_version_id="73f7a60c-be3f-45fa-b83b-dd0e3efff629" title="stock_prediction.py" contentType="text/python">

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Function to fetch stock data
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data['Close']

# Function to prepare data for LSTM
def prepare_data(data, look_back=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(len(scaled_data) - look_back):
        X.append(scaled_data[i:(i + look_back), 0])
        y.append(scaled_data[i + look_back, 0])
    
    return np.array(X), np.array(y), scaler

# Create LSTM model
def create_model(look_back):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Main execution
def main():
    # Parameters
    ticker = 'AAPL'  # Apple stock
    look_back = 60   # Number of days to look back
    prediction_days = 30
    
    # Calculate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1000)
    
    # Get data
    data = get_stock_data(ticker, start_date, end_date)
    
    # Prepare data
    X, y, scaler = prepare_data(data, look_back)
    
    # Split data
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Reshape data for LSTM
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    # Create and train model
    model = create_model(look_back)
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)
    
    # Make predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform([y_test])
    
    # Visualize results
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.T, label='Actual Prices')
    plt.plot(predictions, label='Predicted Prices')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    
    # Predict future prices
    last_sequence = X[-1].reshape((1, look_back, 1))
    future_predictions = []
    
    for _ in range(prediction_days):
        next_pred = model.predict(last_sequence)
        future_predictions.append(next_pred[0, 0])
        last_sequence = np.roll(last_sequence, -1)
        last_sequence[0, -1, 0] = next_pred
    
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    
    # Plot future predictions
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(data)), scaler.inverse_transform(X[:, -1, :]), label='Historical Prices')
    plt.plot(range(len(data), len(data) + prediction_days), future_predictions, label='Future Predictions')
    plt.title(f'{ticker} Stock Price Future Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()