import os, yaml, numpy as np
from sklearn.model_selection import train_test_split
p = yaml.safe_load(open("params.yaml"))["preprocess"]
d = np.load("data/raw/fashion_mnist.npz")
x = d["x_train"].astype("float32") / 255.0
xte = d["x_test"].astype("float32") / 255.0
xtr, xval, ytr, yval = train_test_split(x, d["y_train"], test_size=p["test_size"],
                                        random_state=p["seed"], stratify=d["y_train"])
os.makedirs("data/processed", exist_ok=True)
np.savez_compressed("data/processed/data.npz", x_train=xtr, y_train=ytr,
                    x_val=xval, y_val=yval, x_test=xte, y_test=d["y_test"])