import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import joblib
import pandas as pd
model = load_model("lstm_model.keras")
scaler = joblib.load("scaler.save")
df_hourly = pd.read_csv(
    "df_hourly.csv",
    index_col=0,
    parse_dates=True
)
last_48 = df_hourly.iloc[-48:]
last_24_real = last_48.iloc[:24].values
real_next_24 = last_48.iloc[24:].values
last_24_scaled = scaler.transform(
    last_24_real.reshape(-1, 1)
)
forecast_input = last_24_scaled.copy()
future_predictions = []
for _ in range(24):
    input_3d = forecast_input.reshape(
        1,
        forecast_input.shape[0],
        1
    )
    next_scaled = model.predict(
        input_3d,
        verbose=0
    )[0]
    next_value = scaler.inverse_transform(
        next_scaled.reshape(1, -1)
    )[0][0]
    future_predictions.append(next_value)
    forecast_input = np.vstack(
        [
            forecast_input[1:],
            next_scaled.reshape(1, 1)
        ]
    )
plt.figure(figsize=(12, 5))
plt.plot(
    real_next_24.flatten(),
    label="Gerçek (Gelecek 24 Saat)",
    linewidth=2
)
plt.plot(
    future_predictions,
    label="Tahmin (Gelecek 24 Saat)",
    linewidth=2,
    linestyle="--"
)
plt.title("Gelecek 24 Saat Enerji Tüketimi: Gerçek vs Tahmin")
plt.xlabel("Saat")
plt.ylabel("Global Active Power (kW)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()