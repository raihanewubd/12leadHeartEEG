# 12 Lead Heart EEG


## Project Purpose

This repository contains experiments for classifying 12‑lead ECG signals and visualizing the neural network used for the task.  Streamlit apps allow inspecting intermediate layer activations on a subset of the PhysioNet 12‑lead dataset.

## Installation

1. Create a Python 3 environment.
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

The repository expects a pickled PyTorch dataset named `physionet_subset_dataset_dill.pkl`.
A small subset (about 40 samples) can be generated with:

```bash
python load_dataset.py
```

This script loads the full PhysioNet dataset, randomly samples a few examples per label and saves the result as `physionet_subset_dataset_dill.pkl`.  Alternatively you may place an existing pickle with that name in the project directory.

## Running the Streamlit Apps

Several Streamlit applications are provided to visualize the model and dataset. Run any of them with:

```bash
streamlit run streamlit_app.py
```

Replace `streamlit_app.py` with `streamlit_app03.py` or `streamlit_app04.py` to try alternative interfaces.


Each app loads `everyEpochModelv1.pth` and the dataset pickle, then allows selecting a sample and exploring the feature maps of each neural network layer.
