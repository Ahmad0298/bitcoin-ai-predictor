
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
            # Here we will integrate the LSTM model for prediction
            # For now, just printing the price

# This part will be integrated into the main application loop
# ws = websocket.WebSocketApp(SOCKET, on_open=on_open, on_close=on_close, on_message=on_message)
# ws.run_forever()


# main.py - Real-time Bitcoin Price Predictor

# This file will contain the core logic for data acquisition, model inference, and output.
# Further development is required to implement the real-time prediction logic.




# --- Data Preparation for LSTM --- #

def prepare_data(historical_data):
    # This function will preprocess historical data for LSTM training.
    # It will involve:
    # 1. Normalization/Scaling
    # 2. Creating sequences (e.g., look-back periods)
    # 3. Splitting into training and testing sets
    print("Preparing historical data...")
    # Placeholder for actual data preparation logic
    return historical_data # For now, just return as is

# Placeholder for historical data download/loading
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
