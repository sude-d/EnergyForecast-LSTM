import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, Input
from keras.callbacks import EarlyStopping
from keras.losses import MeanSquaredError
import numpy as np
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")
model = Sequential()
model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
model.add(
    LSTM(
        64,
        activation="tanh"
    )
)
model.add(Dense(1))
model.compile(
    optimizer="adam",
    loss=MeanSquaredError()
)
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)
plt.figure(figsize=(10, 5))
plt.plot(
    history.history["loss"],
    label="Eğitim Kaybı"
)
plt.plot(
    history.history["val_loss"],
    label="Doğrulama Kaybı"
)
plt.title("Model Kayıp Grafiği")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
model.save("lstm_model.keras")