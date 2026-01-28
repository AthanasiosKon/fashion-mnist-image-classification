import tensorflow as tf
from tensorflow import keras
import numpy as np


def load_data(validation_size=5000):
    """
    Loads the Fashion MNIST dataset, normalizes it, and splits it
    into training, validation, and test sets.

    Parameters
    ----------
    validation_size : int   (Number of samples to use for validation.)

    Returns
    -------
    X_train, y_train, X_valid, y_valid, X_test, y_test
    """

    # Load dataset
    fashion_mnist = keras.datasets.fashion_mnist
    (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

    # Normalize pixel values
    X_train_full = X_train_full.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    # Split training and validation sets
    X_valid = X_train_full[:validation_size]
    X_train = X_train_full[validation_size:]

    y_valid = y_train_full[:validation_size]
    y_train = y_train_full[validation_size:]

    return X_train, y_train, X_valid, y_valid, X_test, y_test
