import numpy as np
import matplotlib.pyplot as plt
from model import build_model
from data import load_data

# -------------------------------
# Data and model loading
# -------------------------------
X_train, y_train, X_valid, y_valid, X_test, y_test = load_data()

model = build_model()
model.load_weights("model_weights.h5")  # αν έχεις αποθηκεύσει βάρη, αλλιώς τρέξε train.py πρώτα

# -------------------------------
# Class names definition
# -------------------------------
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# -------------------------------
# Batch parameters
# -------------------------------
batch_size = 9
num_batches = int(np.ceil(len(X_test) / batch_size))

# -------------------------------
# Demo loop for every batch
# -------------------------------
for batch_idx in range(num_batches):
    start = batch_idx * batch_size
    end = start + batch_size
    X_batch = X_test[start:end]

    y_proba = model.predict(X_batch)
    y_pred_idx = np.argmax(y_proba, axis=1)

    plt.figure(figsize=(10, 10))
    for i in range(len(X_batch)):
        plt.subplot(3, 3, i + 1)
        plt.imshow(X_batch[i], cmap='gray')
        plt.axis('off')

        # Top-3 predicted classes
        top3_idx = np.argsort(y_proba[i])[::-1][:3]
        top3_text = "\n".join([f"{class_names[j]} ({y_proba[i][j]:.2f})" for j in top3_idx])
        plt.title(top3_text, fontsize=8)

    plt.tight_layout()
    plt.show()

    user_input = input("Press Enter for the next batch or 'q' to exit: ")
    if user_input.lower() == 'q':
        print("Demo stopped.")
        break
