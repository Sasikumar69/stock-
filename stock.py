import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Step 1: Load stock data (e.g., Apple)
stock = 'AAPL'
df = yf.download(stock, start='2015-01-01', end='2023-12-31')
df = df[['Close']]

# Step 2: Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Step 3: Prepare training data
def create_dataset(data, time_step=60):
    X, Y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i - time_step:i, 0])
        Y.append(data[i, 0])
    return np.array(X), np.array(Y)

time_step = 60
X, Y = create_dataset(scaled_data, time_step)

X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # LSTM input shape

# Step 4: Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 5: Train the model
model.fit(X, Y, batch_size=64, epochs=10)

# Step 6: Predict and visualize
train_predict = model.predict(X)
train_predict = scaler.inverse_transform(train_predict)
Y_true = scaler.inverse_transform(Y.reshape(-1, 1))

# Plot the results
plt.figure(figsize=(14, 6))
plt.plot(df.index[time_step:], Y_true, label="Actual Price")
plt.plot(df.index[time_step:], train_predict, label="Predicted Price")
plt.title(f"{stock} Stock Price Prediction")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()
