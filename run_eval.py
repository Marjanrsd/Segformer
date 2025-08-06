import csv
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torchio as tio
import torch.optim as optim
from datetime import datetime
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings("ignore", category=UserWarning, 
                        module="torchio.data.image")

class SliceSpecificLeNet(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, stride=1, padding=2),  # -> [B, 6, 128, 128]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # -> [B, 6, 64, 64]
            nn.Conv2d(6, 16, kernel_size=5),  # -> [B, 16, 60, 60]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # -> [B, 16, 30, 30]
        )
        self.fc = nn.Sequential(
            nn.Flatten(),                  # -> [B, 16*30*30]
            nn.Linear(16 * 30 * 30, 40),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)  # -> [B, 40]
        return x

# We are basically running each slice of all people in one batch in parallel, 
# and at the end we concatenate them and pass through MLP

class MRIStackedModel(nn.Module):
    def __init__(self, num_slices=16, mlp_hidden_dim=256, output_dim=68):
        super().__init__()
        self.num_slices = num_slices
        
        # Each slice has its own ResNet18
        s = [SliceSpecificLeNet() for _ in range(num_slices)]
        self.slice_nets = nn.ModuleList(s)

        # MLP: Two FC layers
        self.mlp = nn.Sequential(
            nn.Linear(40 * num_slices, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: [B, C, D, H, W]
        B, C, D, H, W = x.shape
        assert D == self.num_slices, f"Expected {self.num_slices} slices, got {D}"

        features = []
        for d in range(D):
            slice_d = x[:, :, d, :, :]           # [B, C, H, W]
            feat_d = self.slice_nets[d](slice_d) # [B, 40]
            features.append(feat_d)

        x_cat = torch.cat(features, dim=1)        # [B, 40 * D]
        output = self.mlp(x_cat)                  # [B, 2]
        return output

class MRIDataset(Dataset):
    def __init__(self, split='train', deg=0):        
        # use train dataset to normalize (z-score) & center data
        # these 4 parameters are potentially overwritten later
        self.data = pd.read_csv(f"/mnt/chrastil/users/marjanrsd/openbhb_ct/old_train.csv")
        self.split = 'train'
        self.deg = 0
        #'''
        self.scans_mean = None
        scans = []
        labels = []
        for i in range(len(self.data)):
            scan, label = self[i]
            scans.append(scan.flatten())
            labels.append(label)
        scans = np.stack(scans, axis=0)
        labels = np.stack(labels, axis=0)
        # for z-scoring & centering about 0
        self.scans_mean = scans.mean()
        self.scans_std = scans.std()
        self.labels_min = labels.min(axis=0)
        self.labels_max = labels.max(axis=0)
        #print((self.labels_max - self.labels_min).mean())
        #'''
        # the actual parameter values are set after pre-processing
        self.data = pd.read_csv(f"/mnt/chrastil/users/marjanrsd/openbhb_ct/{split}.csv")
        self.split = split
        self.deg = deg # data aug rotation amount

    def __len__(self):
        return len(self.data)
    
    def set_deg(self, d=0):
        self.deg = d

    def __getitem__(self, idx):
        # Get the path to the numpy file
        npy_path = f"/mnt/chrastil/users/marjanrsd/openbhb_ct/{self.data.iloc[idx, 0]}" # t1_path column (first col)
        voxel_data = np.load(npy_path, allow_pickle=True) # Load 3D MRI scan

        # Get the labels (ROIs avg thickness values)
        labels = self.data.iloc[idx, 1:]
        labels = torch.tensor(labels, dtype=torch.float32)

        # Convert voxel data to tensor (float32 for neural networks)
        voxel_tensor = torch.tensor(voxel_data, dtype=torch.float32)

        if self.split == "train" and self.deg != 0:
            # Rotates up to ±deg degrees
            random_rotation = tio.RandomAffine(scales=0., degrees=self.deg)
            rotated_T1 = random_rotation(voxel_tensor)
            # set negative values to 0
            rotated_T1[rotated_T1 < 0] = 0.
            voxel_tensor = rotated_T1

        voxel_tensor = voxel_tensor.squeeze()
        if self.scans_mean != None:
            voxel_tensor = (voxel_tensor - self.scans_mean) / self.scans_std
            labels = (labels - self.labels_min) / (self.labels_max - self.labels_min)

        return voxel_tensor, labels



# Testing function
def test_epoch(model, loader, criterion, device=None):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for voxels, labels in loader:
            voxels, labels = voxels.to(device), labels.to(device)
            outputs = model(voxels.unsqueeze(1))
            for batch_i in range(outputs.shape[0]):
                _labels = list(labels[batch_i].cpu().numpy())
                _outputs = list(outputs[batch_i].cpu().numpy())
                for l, o in zip(_labels, _outputs):
                    print(f'l: {l}, o: {o}')
                    break
            loss = criterion(outputs.squeeze(), labels)
            total_loss += loss.item()
    
    return total_loss / len(loader)

if __name__ == "__main__":
    # Load datasets
    test_data = MRIDataset(split="test")
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False, num_workers=8)

    # Model, loss, optimizer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MRIStackedModel(num_slices=128)
    model.load_state_dict(torch.load("best_openbhb.pth", map_location=device))
    model.to(device)
    criterion = nn.MSELoss()
    model.eval()
    

    # Training loop
    best_loss = float("inf") # Initialize best loss as infinity
    test_losses = []
    
    dt_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
   
    test_loss = test_epoch(model, test_loader, criterion, device)
    test_losses.append(test_loss)
    print(f"Test Loss: {test_loss:.4f}")