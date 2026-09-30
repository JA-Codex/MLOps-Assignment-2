import os, json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow import keras
d = np.load("data/processed/data.npz")
m = keras.models.load_model("models/model.h5")
loss, acc = m.evaluate(d["x_test"], d["y_test"], verbose=0)
pred = m.predict(d["x_test"], verbose=0).argmax(1)
os.makedirs("reports", exist_ok=True)
ConfusionMatrixDisplay(confusion_matrix(d["y_test"], pred)).plot(cmap="Blues")
plt.savefig("reports/confusion_matrix.png", dpi=120)
json.dump({"test_loss": float(loss), "test_accuracy": float(acc)}, open("metrics.json", "w"), indent=2)