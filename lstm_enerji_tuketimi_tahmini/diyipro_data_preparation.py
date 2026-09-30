import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import joblib
df_hourly = pd.read_csv(
    "df_hourly.csv",
    index_col=0,
    parse_dates=True
)
df_hourly.dropna(inplace=True)
values = df_hourly.values.reshape(-1, 1)
scaler = MinMaxScaler()
scaled = scaler.fit_transform(values)
joblib.dump(scaler, "scaler.save")
def create_sliding_window(data, window_size=24):
    X = []
    y = []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)
window_size = 24
X, y = create_sliding_window(scaled, window_size)
split = int(len(X) * 0.8)
X_train = X[:split]
X_test = X[split:]
y_train = y[:split]
y_test = y[split:]
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)