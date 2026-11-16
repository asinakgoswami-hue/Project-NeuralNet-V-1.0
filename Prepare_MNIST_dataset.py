'''
Code Property of: Asinak Goswami (GitHub username: asinakgoswami-hue)

Date of push: 16/11/2026 (DD/MM/YYYY)

This is a learning implementation. Feel free to add comments on what I should work on next!

'''

# ======================================== ACTUAL CODE ======================================== 

"""
Prepare_MNIST.py
----------------
Downloads the official MNIST dataset (via scikit-learn) and saves it
as NumPy arrays in the exact layout expected by all the other scripts
in this repository.

Output files (created in the current working directory):
    X_train.npy   shape (784, 60000)
    Y_train.npy   shape (10, 60000)   – one-hot encoded
    X_test.npy    shape (784, 10000)
    Y_test.npy    shape (10, 10000)   – one-hot encoded
    y_test.npy    shape (10000,)      – integer labels (convenient for accuracy)
"""

import numpy as np
from sklearn.datasets import fetch_openml

print("Downloading MNIST...")

X, y = fetch_openml(
    "mnist_784",
    version=1,
    return_X_y=True,
    as_frame=False
)

print("Download complete.")
print("Processing data...")

# Normalize pixels to the range [0, 1]
X = X.astype("float32") / 255.0

# Official train / test split used by almost every MNIST paper
X_train = X[:60000]
y_train = y[:60000]

X_test = X[60000:]
y_test = y[60000:]

# Convert labels to integers
y_train = y_train.astype(int) # type: ignore
y_test = y_test.astype(int) # type: ignore

# Transpose so that columns = examples
# (this is the layout used by the fully-connected networks)
X_train = X_train.T
X_test = X_test.T

# One-hot encode the labels
Y_train = np.eye(10, dtype=np.float32)[y_train].T
Y_test = np.eye(10, dtype=np.float32)[y_test].T

# Save everything
np.save("X_train.npy", X_train)
np.save("Y_train.npy", Y_train)
np.save("X_test.npy", X_test)
np.save("Y_test.npy", Y_test)
np.save("y_test.npy", y_test)

print("MNIST successfully saved.")
