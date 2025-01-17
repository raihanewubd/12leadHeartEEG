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
import pickle
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


root_dir = r'F:\12leadECG\WFDBRecords'
#dataset = PhysioNetDataset(root_dir, start_folder=1, end_folder=46)

end_time = time.time()
loading_time = end_time - start_time
print(f"Dataset Loading Time: {loading_time:.2f} seconds ---> {loading_time/60:.2f} minutes")

# Save the dataset object
#start_time = time.time()
#with open('physionet_dataset.pkl', 'wb') as f:
#    pickle.dump(dataset, f)
#print("Dataset object saved successfully.")
#end_time = time.time()
#loading_time = end_time - start_time
#print(f"Dataset object saveing Time: {loading_time:.2f} seconds ---> {loading_time/60:.2f} minutes")

start_time = time.time()
with open(r'physionet_dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)
print("Dataset object loaded successfully.")
end_time = time.time()
loading_time = end_time - start_time

dataset_size = len(dataset)
print(f"Total dataset size: {dataset_size}")

class_counts = {'AFIB': 0, 'GSVT': 0, 'SB': 0, 'SR': 0}
for _, label in dataset:
    for class_name, class_index in {'AFIB': 0, 'GSVT': 1, 'SB': 2, 'SR': 3}.items():
        if label.item() == class_index:
            class_counts[class_name] += 1
            break

print()
for class_name, count in class_counts.items():
    print(f"{class_name}: {count}")
print(f"Dataset object loading Time: {loading_time:.2f} seconds ---> {loading_time/60:.2f} minutes")
print(len(dataset))
sample_idx = 10
signal, label = dataset[sample_idx]

print("Signal Shape:", signal.shape)
print("Signal Data:", signal)
print("Label:", label.item())

train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(seed_value)
)

print(f"Train set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")
print(f"Test set size: {len(test_dataset)}")

def print_class_distribution(dataset, dataset_name):
    class_counts = defaultdict(int)
    for _, label in dataset:
        class_counts[label.item()] += 1

    class_map = {0: 'AFIB', 1: 'GSVT', 2: 'SB', 3: 'SR'}
    total_samples = sum(class_counts.values())

    print(f"Class distribution for {dataset_name}:")
    for label, count in class_counts.items():
        print(f"  {class_map[label]}: {count} samples")

    print(f"  Total samples in {dataset_name}: {total_samples}\n")

print_class_distribution(train_dataset, "Training Set")
print_class_distribution(val_dataset, "Validation Set")
print_class_distribution(test_dataset, "Test Set")
train_batch = 32
val_test_batch = 32

train_loader = DataLoader(train_dataset, batch_size=train_batch, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=val_test_batch, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=val_test_batch, shuffle=False)



seed_value = 42
torch.manual_seed(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

torch.cuda.manual_seed(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False