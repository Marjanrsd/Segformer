import os
import torch
import numpy as np
import torchio as tio
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import SegformerConfig, SegformerForSemanticSegmentation

try:
    import cupy as cp
    from cupyx.scipy.ndimage import zoom
    cuda_available = True
except:
    from scipy.ndimage import zoom
    cuda_available = False  

import warnings
warnings.filterwarnings("ignore", category=UserWarning, 
                        module="torchio.data.image")

total_chs = 1 # total number of channels: 2 * padding_chs + 1
# the number of extra slices on each side of the central slice
padding_chs = (total_chs - 1) // 2
target_size = 160 # assumes cubic shape
start_deg = 0 # used for data-aug rotation

class MRIDataset(Dataset):
    def __init__(self, split='train', viz=False, deg=0):
        self.split = split 
        self.viz = viz
        self.deg = deg # data aug rotation amount
        self.transform = tio.Compose([
            tio.RandomAffine(scales=(0.8, 1.2), degrees=self.deg),
            tio.RandomFlip(axes=(0, 1, 2)),
        ])
        # sanity check
        data_dir = f'seg_labels/{self.split}'
        all_files = os.listdir(data_dir)
        voxel_files = [f for f in all_files if f.endswith("_T1.npy")]
        label_files = [f for f in all_files if f.endswith("_labels.npy")]
        err_txt = "number of voxels files != number of label files"
        assert len(voxel_files) == len(label_files), err_txt
        voxel_files = [os.path.join(data_dir, f) for f in voxel_files]
        self.voxel_files = voxel_files

    def __len__(self):
        return len(self.voxel_files)

    def __getitem__(self, idx):
        npy_path = self.voxel_files[idx]
        label_path = npy_path.replace("T1", "labels")
        assert label_path != npy_path, f"{npy_path} has incorrect naming format"
        voxel_data = np.load(npy_path)
        label_data = np.load(label_path) # [192,192,192]
        voxel_tensor = torch.tensor(voxel_data, dtype=torch.float32)
        label_tensor = torch.tensor(label_data, dtype=torch.long)  # [1, D, H, W]
        
        if self.deg != 0:
            voxel_tensor = voxel_tensor.unsqueeze(0)
            label_tensor = label_tensor.unsqueeze(0)
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=voxel_tensor),
                label=tio.LabelMap(tensor=label_tensor)
            )
            transformed = self.transform(subject)
            # Access the transformed tensors
            voxel_tensor = transformed['image'].data
            label_tensor = transformed['label'].data
            voxel_tensor = voxel_tensor.squeeze(0)
            label_tensor = label_tensor.squeeze(0)            

        if self.viz:
            return voxel_tensor, label_tensor
        else:
            max_bgnd = 0.6 # max allowed percentage of background
            bgnd_perc = 1.0 # ensures while loop is entered
            min_cl = 6 # min num of unique brain regions / classes
            num_cl = 0 # number of unique ROIs in the label slice
            # keep grabbing slices until 1 has enough non-background & unique ROIs
            while bgnd_perc > max_bgnd and num_cl < min_cl:
                slice_range = voxel_tensor.shape[0] - (2 * padding_chs)
                # Select central slice
                if self.split == "train":
                    central_i = int(np.random.random() * slice_range) + padding_chs
                else: # test always uses the middle slice
                    central_i = voxel_tensor.shape[0]//2 + padding_chs
                slice_idxs = list(range(central_i-padding_chs, central_i+padding_chs+1))
                image_slice = voxel_tensor[slice_idxs]
                bgnd_perc = (voxel_tensor[central_i] == 0).sum().float() / voxel_tensor[central_i].numel()
                label_slice = label_tensor[central_i]
                num_cl = len(torch.unique(label_slice))
                if self.split == "test":
                    break # only run once 

        image_slice = image_slice / 255.0

        # image_slice is [C, H, W], so add a fake batch dim:
        image_slice = image_slice.unsqueeze(0)  # [1, C, H, W]
        image_slice = F.interpolate(
            image_slice,
            size=(target_size, target_size),
            mode='bilinear',
            align_corners=False
        )
        image_slice = image_slice.squeeze(0)  # back to [C, target_h, target_w]

        # (optional) if you want to resize the label map too:
        label_slice = label_slice.unsqueeze(0).unsqueeze(0).float() # [1,1,H,W]
        label_slice = F.interpolate(label_slice, size=(target_size,target_size),
                                    mode='nearest')
        label_slice = label_slice.long().squeeze(0).squeeze(0) # [target_h,target_w]

        return image_slice, label_slice

# === Hook setup ===
intermediate_output = {}

def hook_fn(module, input, output):
    # Save fused decoder features before final classifier
    intermediate_output['decoder_output'] = output.detach().cpu().numpy()


if __name__ == "__main__":
    config = SegformerConfig(
        num_labels=61,
        image_size=target_size,
        num_channels=total_chs,
        patch_sizes=[7, 3, 3, 3],
        strides=[1, 2, 2, 2],             # Total downsample = 8
        hidden_sizes=[32, 64, 160, 256], # try [64, 128, 320, 512],
        depths=[2, 2, 2, 2],             # try [3, 4, 6, 3]
        mlp_ratios=[4, 4, 4, 4],
        decoder_hidden_size=256,
        reshape_last_stage=True,
    )

    model = SegformerForSemanticSegmentation(config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # load pretrained model
    save_path = "recreate_1ch_180deg.pth"
    load = input(f'load {save_path}? [y/n] ')
    load = 'y' in load
    if load:
        try:
            state_dict = torch.load(save_path)
            model.load_state_dict(state_dict["model_state_dict"])
        except:
            print("ERROR loading previous model!") 

     # Register the hook to the decoder's linear_fuse layer
    hook_handle = model.decode_head.linear_fuse.register_forward_hook(hook_fn)

    test_data = MRIDataset(split="train")
    #test_data = MRIDataset(split="test")
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=6)

    # === Save decoder output for all test datapoints ===
    output_dir = "decoder_features"
    os.makedirs(output_dir, exist_ok=True)

    for idx, (voxels, labels) in enumerate(test_loader):
        voxels = voxels.to(device)
        with torch.no_grad():
            _ = model(pixel_values=voxels)

        feature = intermediate_output['decoder_output']  # (1, 256, H, W)
        feature = feature[0].transpose(1, 2, 0)           # (H, W, 256)

        # Get original T1 filename ===
        t1_path = test_data.voxel_files[idx]
        t1_filename = os.path.basename(t1_path)  # e.g., "0001_T1.npy"
        
        # === Save with the same filename ===
        save_path = os.path.join(output_dir, t1_filename)
        np.save(save_path, feature)

        print(f"Saved {save_path}")

    print("All intermediate outputs saved.")
    
