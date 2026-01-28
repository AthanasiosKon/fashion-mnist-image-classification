import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data import load_data
from model import build_model

def main():
    tf.random.set_seed(42)
    np.random.seed(42)

    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data()
    model = build_model()
    
    # model inspection
    model.summary()
    first_hidden_layer = model.layers[1]
    weights, biases = first_hidden_layer.get_weights()

    history = model.fit(
        X_train,
        y_train,
        epochs=30,
        validation_data=(X_valid, y_valid)
    )

    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.ylim(0, 1)
    plt.show()

    model.evaluate(X_test, y_test)
    model.save("model.h5")

if __name__ == "__main__":
    main()
