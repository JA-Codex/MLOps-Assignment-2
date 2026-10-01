import os, numpy as np
from tensorflow import keras
(xtr, ytr), (xte, yte) = keras.datasets.fashion_mnist.load_data()
os.makedirs("data/raw", exist_ok=True)
np.savez_compressed("data/raw/fashion_mnist.npz", x_train=xtr, y_train=ytr, x_test=xte, y_test=yte)

