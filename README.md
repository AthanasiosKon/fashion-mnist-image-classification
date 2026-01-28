# Fashion MNIST Image Classification

This project classifies images from the **Fashion MNIST** dataset using a neural network built with **TensorFlow** and **Keras**. It demonstrates loading and preprocessing the data, building and training a model, and visualizing predictions interactively.

## Project Structure

- `src/data.py` – Loads and preprocesses the dataset.
- `src/model.py` – Defines the neural network architecture.
- `src/train.py` – Trains the model and saves the trained weights.
- `src/demo.py` – Loads the trained weights and displays batches of test images with the top-3 predicted classes.

## Installation

Clone this repository to your local machine:

    git clone https://github.com/AthanasiosKon/fashion-mnist-image-classification.git
    cd fashion-mnist-image-classification

Create a virtual environment to manage dependencies. Using **Anaconda**:

    conda create -n tf310 python=3.10
    conda activate tf310
    pip install -r requirements.txt

## Usage

Train the model and save the weights:

    python src/train.py

Run the demo to visualize predictions:

    python src/demo.py

The demo displays batches of images (default 9 per batch). Press Enter to see the next batch. Each image shows the top-3 predicted classes with probabilities.

## Dependencies

- Python 3.10
- TensorFlow 2.20+
- Keras
- NumPy
- Pandas
- Matplotlib

All dependencies are listed in `requirements.txt`.

## License

This project is released under the MIT License.
