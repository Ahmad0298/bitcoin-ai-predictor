


from binance.client import Client
import datetime

# --- Historical Data Fetching --- #

# IMPORTANT: Replace with your actual Binance API Key and Secret
# For security, it is highly recommended to load these from environment variables
# or a secure configuration management system, NOT hardcoded in the file.
API_KEY = "YOUR_BINANCE_API_KEY"
API_SECRET = "YOUR_BINANCE_API_SECRET"

def fetch_historical_klines(symbol, interval, start_str, end_str=None):
    client = Client(API_KEY, API_SECRET)
    all_klines = []
    # Binance API has a limit of 1000 klines per request. We need to loop to get more.
    # For 1-minute interval, 1 year is 365 * 24 * 60 = 525600 klines. This requires many requests.
    # For second-by-second prediction, historical 1-second data is not directly available via REST API.
    # We will fetch 1-minute data and then potentially interpolate or use it for training.

    start_ts = Client.get_timestamp(start_str)
    end_ts = Client.get_timestamp(end_str) if end_str else int(datetime.datetime.now().timestamp() * 1000)

    while True:
        # Fetch klines in chunks
        klines = client.get_historical_klines(symbol, interval, start_ts, end_ts, limit=1000)
        if not klines:
            break
        all_klines.extend(klines)
        # Set new start_ts to the timestamp of the last fetched kline + 1 millisecond
        start_ts = klines[-1][0] + 1
        # Break if we have fetched up to the end_ts
        if start_ts > end_ts:
            break

    # Extract close prices from the klines
    historical_prices = [float(k[4]) for k in all_klines]
    print(f"Fetched {len(historical_prices)} historical klines.")
    return historical_prices

import websocket
import json
import time

# Binance WebSocket API endpoint for Bitcoin (BTC/USDT) 1-second klines
SOCKET = "wss://stream.binance.com:9443/ws/btcusdt@kline_1s"

def on_open(ws):
    print("Opened connection")

def on_close(ws):
    print("Closed connection")



# --- Real-time Prediction Integration --- #

# Global variables to store the trained model and scaler
trained_model = None
price_scaler = None
recent_prices = [] # To store the last 'look_back' prices for prediction
LOOK_BACK = 60 # This should match the look_back used in prepare_data


def load_and_train_model():
    global trained_model, price_scaler
    print("Loading/Training model...")
    # Fetch historical data (e.g., 1 year of daily data for initial training)
    # You might need to adjust the interval and start_str based on your needs
    # For real-time prediction, 1-minute or 5-minute intervals might be more suitable
    # for training, and then predicting 1-second changes.
    try:
        historical_prices = fetch_historical_klines(symbol='BTCUSDT', interval=Client.KLINE_INTERVAL_1MINUTE, start_str='1 Jan, 2024')
        if not historical_prices:
            print("No historical data fetched. Using dummy data for now.")
            historical_prices = [i for i in range(100, 200)] # Fallback to dummy
    except Exception as e:
        print(f"Error fetching historical data: {e}. Using dummy data for now.")
        historical_prices = [i for i in range(100, 200)] # Fallback to dummy

    # Now, use this historical_prices for data preparation and model training
    X, y, price_scaler = prepare_data(historical_prices, look_back=LOOK_BACK)

    # Split data into training and testing sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[0:train_size,:], X[train_size:len(X),:]
    y_train, y_test = y[0:train_size], y[train_size:len(y)]

    # Build and train the model
    model = build_lstm_model(input_shape=(X_train.shape[1], 1))
    trained_model = train_and_evaluate_model(model, X_train, y_train, X_test, y_test)

    # Save the trained model (optional, but recommended for persistence)
    trained_model.save('bitcoin_lstm_model.h5')
    print("Trained model saved as bitcoin_lstm_model.h5")
    print("Model and scaler initialized.")



def on_message(ws, message):
    global trained_model, price_scaler, recent_prices

    if trained_model is None or price_scaler is None:
        print("Model or scaler not initialized. Skipping prediction.")
        return

    json_message = json.loads(message)

    if 'k' in json_message and 'c' in json_message['k']:
        candle = json_message['k']
        is_candle_closed = candle['x']
        close_price = float(candle['c'])

        if is_candle_closed:
            print(f"Current BTC Price: {close_price}")

            # Add current price to recent_prices and maintain LOOK_BACK size
            recent_prices.append(close_price)
            if len(recent_prices) > LOOK_BACK:
                recent_prices.pop(0)

            if len(recent_prices) == LOOK_BACK:
                # Prepare data for prediction
                last_60_prices = np.array(recent_prices).reshape(-1, 1)
                scaled_last_60_prices = price_scaler.transform(last_60_prices)
                X_predict = np.array([scaled_last_60_prices])
                X_predict = np.reshape(X_predict, (X_predict.shape[0], X_predict.shape[1], 1))

                # Make prediction
                predicted_scaled_price = trained_model.predict(X_predict)[0][0]
                predicted_price = price_scaler.inverse_transform([[predicted_scaled_price]])[0][0]

                print(f"Predicted Next BTC Price: {predicted_price:.2f}")
                if predicted_price > close_price:
                    print("Trend: UP")
                else:
                    print("Trend: DOWN")
            else:
                print(f"Collecting historical data... ({len(recent_prices)}/{LOOK_BACK})")

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
    print("Training LSTM model...")
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1) # Example epochs and batch_size
    print("Model training complete.")

    print("Evaluating LSTM model...")
    loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"Model evaluation loss: {loss:.4f}")
    return model

# Example usage (requires historical_data to be loaded and prepared)
# For demonstration, we'll use the dummy historical data
# In a real scenario, you would load actual historical data

# Assuming historical_prices_dummy is available from data preparation section
# X, y, scaler = prepare_data(historical_prices_dummy)

# Split data into training and testing sets (simple split for demonstration)
# train_size = int(len(X) * 0.8)
# X_train, X_test = X[0:train_size,:], X[train_size:len(X),:]
# y_train, y_test = y[0:train_size], y[train_size:len(y)]

# model = build_lstm_model(input_shape=(X_train.shape[1], 1)) # input_shape should be (timesteps, 1)
# trained_model = train_and_evaluate_model(model, X_train, y_train, X_test, y_test)


# X_train, y_train, X_test, y_test = prepare_data(historical_data)
# model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
# trained_model = train_and_evaluate_model(model, X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    ws = websocket.WebSocketApp(SOCKET, on_open=on_open, on_close=on_close, on_message=on_message)
        load_and_train_model()
ws.run_forever()
