
import websocket
import json
import time

# Binance WebSocket API endpoint for Bitcoin (BTC/USDT) 1-second klines
SOCKET = "wss://stream.binance.com:9443/ws/btcusdt@kline_1s"

def on_open(ws):
    print("Opened connection")

def on_close(ws):
    print("Closed connection")

def on_message(ws, message):
    json_message = json.loads(message)
    # print(json_message)

    # Extract relevant data (e.g., close price)
    if 'k' in json_message and 'c' in json_message['k']:
        candle = json_message['k']
        is_candle_closed = candle['x']
        close_price = float(candle['c'])

        if is_candle_closed:
            print(f"Current BTC Price: {close_price}")

            # Placeholder for actual prediction logic
            # You would typically feed a sequence of recent prices to your trained LSTM model
            # and get a prediction for the next second.
            # For demonstration, let's simulate a prediction.
            predicted_price = close_price * (1 + (0.0001 * (2 * (0.5 - (time.time() % 1))))) # Simple fluctuating prediction
            
            print(f"Predicted Next BTC Price: {predicted_price:.2f}")
            if predicted_price > close_price:
                print("Trend: UP")
            else:
                print("Trend: DOWN")

            # Here we will integrate the LSTM model for prediction
            # For now, just printing the price

# This part will be integrated into the main application loop
# ws = websocket.WebSocketApp(SOCKET, on_open=on_open, on_close=on_close, on_message=on_message)
# ws.run_forever()


# main.py - Real-time Bitcoin Price Predictor

# This file will contain the core logic for data acquisition, model inference, and output.
# Further development is required to implement the real-time prediction logic.





import numpy as np
from sklearn.preprocessing import MinMaxScaler

# --- Data Preparation for LSTM --- #

def prepare_data(data, look_back=60):
    # data: a list or numpy array of historical prices
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(np.array(data).reshape(-1, 1))

    X = []
    y = []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler

# Placeholder for historical data loading
# In a real scenario, you would fetch historical data from an API.
# For now, let's simulate some historical data for demonstration.
# This should be replaced with actual data fetching.
# For example, using python-binance library to get historical klines
# from binance.client import Client
# client = Client(api_key, api_secret)
# klines = client.get_historical_klines(symbol='BTCUSDT', interval=Client.KLINE_INTERVAL_1MINUTE, start_str='1 Jan, 2023')
# historical_prices = [float(k[4]) for k in klines] # Close prices

# For now, let's use a dummy historical data for testing the structure
historical_prices_dummy = [i for i in range(100, 200)] # Simulate 100 data points


# In a real scenario, you would download historical BTC price data
# from an exchange API (e.g., Binance historical klines) or load from a file.
# For demonstration, we'll assume `historical_data` is available.
# historical_data = load_historical_data()



# --- LSTM Model Definition --- #

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1)) # Output a single price prediction
    model.compile(optimizer='adam', loss='mean_squared_error')
    print("LSTM model built.")
    return model

# Placeholder for model training
# model = build_lstm_model(input_shape=(timesteps, features))
# model.fit(X_train, y_train, epochs=..., batch_size=...)



# --- Model Training and Evaluation --- #

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    print("Training and evaluating LSTM model...")
    # Placeholder for actual training and evaluation logic
    # model.fit(X_train, y_train, epochs=..., batch_size=...)
    # evaluation_results = model.evaluate(X_test, y_test)
    # print(f"Model evaluation results: {evaluation_results}")
    return model

# Example usage (requires historical_data to be loaded and prepared)
# X_train, y_train, X_test, y_test = prepare_data(historical_data)
# model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
# trained_model = train_and_evaluate_model(model, X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    ws = websocket.WebSocketApp(SOCKET, on_open=on_open, on_close=on_close, on_message=on_message)
    ws.run_forever()
