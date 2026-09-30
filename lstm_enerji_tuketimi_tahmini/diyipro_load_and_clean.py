import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv(
    "household_power_consumption.txt",
    sep=";",
    na_values="?",
    low_memory=False
)
df["datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    dayfirst=True,
    errors="coerce"
)
df.set_index("datetime", inplace=True)
df["Global_active_power"] = pd.to_numeric(
    df["Global_active_power"],
    errors="coerce"
)
df = df.dropna(subset=["Global_active_power"])
df_hourly = df["Global_active_power"].resample("h").mean()
df_hourly.to_csv("df_hourly.csv")
plt.figure(figsize=(12, 5))
plt.plot(
    df_hourly.index,
    df_hourly.values,
    label="Saatlik Enerji Tüketimi"
)
plt.title("Saatlik Enerji Tüketimi")
plt.xlabel("Zaman")
plt.ylabel("Global Active Power (kW)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()