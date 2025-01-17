import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchinfo import summary

class HRVPooling(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(HRVPooling, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x):
        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding))

        batch_size, channels, length = x.size()

        # Validate input length
        if length < self.kernel_size:
            raise ValueError(f"Input length ({length}) is smaller than kernel size ({self.kernel_size}).")

        # Compute pooled length
        pooled_length = (length - self.kernel_size) // self.stride + 1

        # Extract pooling windows
        x_unfold = x.unfold(dimension=2, size=self.kernel_size, step=self.stride)

        # Compute successive differences within each window
        differences = x_unfold[..., 1:] - x_unfold[..., :-1]
        rmssd = torch.sqrt(torch.mean(differences ** 2, dim=-1) + 1e-8)  # Add epsilon for stability

        # Normalize RMSSD to obtain belief masses
        rmssd_sum = torch.sum(rmssd, dim=-1, keepdim=True) + 1e-8  # Avoid division by zero
        belief_masses = rmssd / rmssd_sum

        # Weighted sum using belief masses
        combined_belief = torch.sum(x_unfold * belief_masses.unsqueeze(-1), dim=-1)

        return combined_belief
class ECGLeadNet(nn.Module):
    def __init__(self):
        super(ECGLeadNet, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=12, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.hrv_pool1 = HRVPooling(kernel_size=5, stride=2)

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.hrv_pool2 = HRVPooling(kernel_size=5, stride=2)

        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(64)
        self.hrv_pool3 = HRVPooling(kernel_size=5, stride=2)

        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.bn4 = nn.BatchNorm1d(128)
        self.hrv_pool4 = HRVPooling(kernel_size=5, stride=2)

        self.adaptive_pool = nn.AdaptiveMaxPool1d(100)
        self.fc1 = nn.Linear(128 * 100, 64)
        self.dropout = nn.Dropout(0.7)
        self.fc2 = nn.Linear(64, 4)

    def forward(self, x):
        internal_outputs = {}  # Dictionary to store outputs at each layer
        x = self.conv1(x)
        internal_outputs['conv1'] = x
        x = self.bn1(F.relu(x))
        internal_outputs['bn1'] = x
        x = self.hrv_pool1(x)
        internal_outputs['hrv_pool1'] = x

        x = self.conv2(x)
        internal_outputs['conv2'] = x
        x = self.bn2(F.relu(x))
        internal_outputs['bn2'] = x
        x = self.hrv_pool2(x)
        internal_outputs['hrv_pool2'] = x

        x = self.conv3(x)
        internal_outputs['conv3'] = x
        x = self.bn3(F.relu(x))
        internal_outputs['bn3'] = x
        x = self.hrv_pool3(x)
        internal_outputs['hrv_pool3'] = x

        x = self.bn4(F.relu(self.conv4(x)))
        internal_outputs['conv4'] = x
        x = self.hrv_pool4(x)
        internal_outputs['hrv_pool4'] = x

        x = self.adaptive_pool(x)
        internal_outputs['adaptive_pool'] = x
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        internal_outputs['fc1'] = x
        x = self.fc2(x)
        internal_outputs['fc2'] = x

        return internal_outputs