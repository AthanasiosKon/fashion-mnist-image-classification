import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

from data import load_data

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

X_train, y_train, X_valid, y_valid, X_test, y_test = load_data()

# visualize samples
plt.figure(figsize=(6, 6))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_train[i], cmap="gray")
    plt.title(class_names[y_train[i]])
    plt.axis("off")
plt.show()

# load trained model
model = keras.models.load_model("model.h5")

# predictions
X_new = X_test[:3]
y_proba = model.predict(X_new)

print(y_proba.round(2))

class_idx = np.argmax(y_proba, axis=1)
results = [class_names[i] for i in class_idx]

print(results)
