import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import wfdb
from scipy.io import loadmat
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import time
from torch.autograd import Function
from sklearn.metrics import mean_squared_error
import re
from sklearn.metrics import r2_score
import dill as pickle
from collections import defaultdict
import random

start_time = time.time()

dx_mapping = {
    '164889003': 'AFIB', '164890007': 'AFIB',
    '427084000': 'GSVT', '426761007': 'GSVT', '713422000': 'GSVT',
    '233896004': 'GSVT', '233897008': 'GSVT', '195101003': 'GSVT',
    '426177001': 'SB',
    '426783006': 'SR', '427393009': 'SR'
}


class PhysioNetDataset(Dataset):
    def __init__(self, root_dir, start_folder=1, end_folder=10):
        self.root_dir = root_dir
        self.data = []
        self.labels = []

        for folder in os.listdir(root_dir):
            try:
                folder_num = int(folder)
            except ValueError:
                continue

            if folder_num < start_folder or folder_num > end_folder:
                continue

            folder_path = os.path.join(root_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            for subfolder in os.listdir(folder_path):
                subfolder_path = os.path.join(folder_path, subfolder)
                if not os.path.isdir(subfolder_path):
                    continue

                for file in os.listdir(subfolder_path):
                    if file.endswith('.hea'):
                        # Read the .hea file
                        record_path = os.path.join(subfolder_path, file[:-4])
                        try:
                            record = wfdb.rdheader(record_path)
                        except Exception as e:
                            print(f"Error reading header for {record_path}: {e}")
                            continue

                        # Extract Dx codes and map to class
                        dx_codes = None
                        for comment in record.comments:
                            if comment.startswith('Dx:'):
                                dx_codes = comment.split(': ')[1].split(',')
                                break

                        if dx_codes is None:
                            print(f"No Dx code found for {record_path}")
                            continue

                        # Map the Dx codes to the appropriate class
                        class_label = None
                        for code in dx_codes:
                            if code.strip() in dx_mapping:
                                class_label = dx_mapping[code.strip()]
                                break

                        if class_label:
                            # Load the signal data from .mat file
                            try:
                                mat_data = loadmat(record_path + '.mat')
                                signal = mat_data['val'].astype(np.float32)
                            except Exception as e:
                                print(f"Error reading .mat file for {record_path}: {e}")
                                continue

                            # Append data and label
                            self.data.append(signal)
                            self.labels.append(class_label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        signal = self.data[idx]
        label = self.labels[idx]

        # Normalize the signal to zero mean and unit variance
        signal = (signal - np.mean(signal)) / np.std(signal)
        signal = torch.tensor(signal).float()

        label_map = {'AFIB': 0, 'GSVT': 1, 'SB': 2, 'SR': 3}
        label = torch.tensor(label_map[label]).long()

        return signal, label
    

def select_n_per_label_randomly(original_dataset, n=10, seed=42):
    """
    From the given dataset, randomly select up to 'n' samples for each integer label,
    and return a new PyTorch Dataset containing only those samples.

    The returned dataset will include the original index of each sample
    in addition to (data, label).

    Args:
        original_dataset: A PyTorch Dataset where each item is (data, label).
        n (int): Number of samples to randomly select per label.
        seed (int): Random seed for reproducibility.
    """
    # For reproducibility (if desired)
    random.seed(seed)
    
    # Accumulate all samples by label
    label_to_samples = defaultdict(list)
    for idx in range(len(original_dataset)):
        data, label = original_dataset[idx]
        label_idx = label.item()  # Convert tensor label to int
        label_to_samples[label_idx].append((data, label, idx))

    # Now randomly sample up to n items from each label
    data_list = []
    label_list = []
    original_indices_list = []

    for label_idx, samples in label_to_samples.items():
        # Shuffle or use random.sample to pick n items
        # random.sample will automatically raise an error if len(samples) < n,
        # so you may want to handle that case if needed.
        selected_samples = random.sample(samples, k=min(n, len(samples)))
        for (data, label, orig_idx) in selected_samples:
            data_list.append(data)
            label_list.append(label)
            original_indices_list.append(orig_idx)

    # Define a simple dataset class that also returns the original index
    class SimpleDataset(Dataset):
        def __init__(self, data_list, label_list, original_indices_list):
            self.data_list = data_list
            self.label_list = label_list
            self.original_indices_list = original_indices_list

        def __len__(self):
            return len(self.data_list)

        def __getitem__(self, i):
            # Return (data, label, original_index)
            return self.data_list[i], self.label_list[i], self.original_indices_list[i]

    return SimpleDataset(data_list, label_list, original_indices_list)


start_time = time.time()
with open(r"E:\Programming\Python\12leadHeartEEG\physionet_dataset_dill.pkl", 'rb') as f:
    dataset = pickle.load(f)
end_time = time.time()
loading_time = end_time - start_time
print(f"Dataset Loading Time: {loading_time:.2f} seconds ---> {loading_time/60:.2f} minutes")
# Assuming 'dataset' is your original PhysioNetDataset instance
start_time = time.time()
subset_dataset = select_n_per_label_randomly(dataset, n=10)
end_time = time.time()
loading_time = end_time - start_time
print(f"select_n_per_label_randomly Time: {loading_time:.2f} seconds ---> {loading_time/60:.2f} minutes")
print(f"New dataset length = {len(subset_dataset)}")

# Inspect a few samples
for i in range(min(40, len(subset_dataset))):
    data, label, orig_idx = subset_dataset[i]
    print(
        f"Sample {i}: data shape = {data.shape}, label = {label.item()}, "
        f"original index = {orig_idx}"
    )

save_path = "physionet_subset_dataset_dill.pkl"
start_time = time.time()
with open(save_path, 'wb') as f:
    pickle.dump(subset_dataset, f)
end_time = time.time()
loading_time = end_time - start_time
print(f"physionet_subset dataset saved to {save_path} save Time: {loading_time:.2f} seconds ---> {loading_time/60:.2f} minutes")
#print(f"physionet_subset dataset saved to {save_path}!")