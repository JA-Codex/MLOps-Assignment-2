import os, yaml, numpy as np, pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
p = yaml.safe_load(open("params.yaml"))["train"]
keras.utils.set_random_seed(p["seed"])
d = np.load("data/processed/data.npz")
model = keras.Sequential([
    keras.Input(shape=(28, 28)),
    layers.Flatten(),
    layers.Dense(p["dense_units"], activation="relu"),
    layers.Dropout(p["dropout_rate"]),
    layers.Dense(10, activation="softmax"),
])
model.compile(optimizer=keras.optimizers.Adam(p["learning_rate"]),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
h = model.fit(d["x_train"], d["y_train"], validation_data=(d["x_val"], d["y_val"]),
              epochs=p["epochs"], batch_size=p["batch_size"])
os.makedirs("models", exist_ok=True)
model.save("models/model.h5")
pd.DataFrame(h.history).to_csv("models/history.csv", index=False)