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
                    central_i = voxel_tensor.shape[1]//2 + padding_chs
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

def visualize_prediction(model, dataloader, device):
    model.eval()

    # Get one 3D volume + label pair
    t1_voxels, labels = next(iter(dataloader))  # t1_voxels: [B, C, H, W], labels: [B, H, W]
    t1_voxels = t1_voxels.to(device).squeeze(0).unsqueeze(1)
    t1_voxels = t1_voxels / 255.0
    t1_voxels = t1_voxels.permute([1,0,2,3])
    _, D, H, W = labels.shape # assumes labels and voxels match size
    scales = (1, target_size / D, target_size / H, target_size / W)

    if cuda_available == True:
        t1_voxels = cp.array(t1_voxels.cpu().numpy())
        labels = cp.array(labels.cpu().numpy())
        with cp.cuda.Device(0): # set on specific GPU
            t1_voxels = zoom(t1_voxels, scales, order=1) # bilinear
            labels = zoom(labels, scales, order=0) # nearest neighbor
        t1_voxels = t1_voxels.squeeze(0)
        t1_voxels = torch.as_tensor(cp.asnumpy(t1_voxels), device=device)
        labels = torch.as_tensor(cp.asnumpy(labels), device=device).squeeze(0)
    else:
        t1_voxels = t1_voxels.cpu().numpy()
        labels = labels.cpu().numpy()
        t1_voxels = zoom(t1_voxels, scales, order=1) # bilinear
        labels = zoom(labels, scales, order=0) # nearest neighbor
        t1_voxels = t1_voxels.squeeze(0)
        t1_voxels = torch.as_tensor(t1_voxels, device=device)
        labels = torch.as_tensor(labels, device=device).squeeze(0)


    blank = torch.zeros_like(t1_voxels[0:1]) # shape: [1, H, W]
    for i in range(padding_chs):
        t1_voxels = torch.cat([blank, t1_voxels, blank], dim=0)
    preds = []
    for i in range(padding_chs, target_size + padding_chs):
        with torch.no_grad():
            if total_chs == 1:
                slices = t1_voxels[i:i+1].unsqueeze(0)
            else:
                slices = t1_voxels[i-padding_chs:i+padding_chs+1].unsqueeze(0)
            logits = model(pixel_values=slices).logits  # (B, num_classes, H, W)
            pred = torch.argmax(logits, dim=1)  # (B, H, W)
            preds.append(pred)
    preds = torch.stack(preds, dim=0).cpu().numpy() # (D, H, W)
    #preds[preds == 25] = 0 # recreate_0.62.pth guesses 25 for a lot of actual bkgn voxels...
    #input('^ acknowledge this hack')
    t1_voxels = t1_voxels.cpu().numpy()
    # remove padding channels
    for i in range(padding_chs):
        t1_voxels = t1_voxels[1:-1]
    labels = labels.cpu().numpy()

    # Initial slice indices for each plane
    axial_idx = target_size // 2 # 160/2 = 80
    sagittal_idx = target_size // 2 # 160/2 = 80
    coronal_idx = target_size // 2 # 160/2 = 80

    fig, axs = plt.subplots(3, 3, figsize=(15, 12))
    plt.subplots_adjust(bottom=0.25, hspace=0.3)

    # Axial (XY plane at depth D)
    slice = t1_voxels[axial_idx]
    ax_img = axs[0, 0].imshow(slice, cmap='gray')
    axs[0, 0].set_title(f'Axial Input Slice {axial_idx}')
    axs[0, 0].axis('off')

    ax_lbl = axs[0, 1].imshow(labels[axial_idx], cmap='tab20', vmin=0, vmax=60)
    axs[0, 1].set_title('Axial Ground Truth')
    axs[0, 1].axis('off')

    slice = np.squeeze(preds[axial_idx,:,:], axis=0)
    ax_pred = axs[0, 2].imshow(slice, cmap='tab20', vmin=0, vmax=60)
    axs[0, 2].set_title('Axial Prediction')
    axs[0, 2].axis('off')

    # Sagittal (YZ plane at width W)
    slice = t1_voxels[:, :, sagittal_idx]
    sag_img = axs[1, 0].imshow(slice, cmap='gray')
    axs[1, 0].set_title(f'Sagittal Input Slice {sagittal_idx}')
    axs[1, 0].axis('off')

    sag_lbl = axs[1, 1].imshow(labels[:, :, sagittal_idx], cmap='tab20', vmin=0, vmax=60)
    axs[1, 1].set_title('Sagittal Ground Truth')
    axs[1, 1].axis('off')

    slice = np.squeeze(preds, axis=1)
    slice = slice[:, :, sagittal_idx]
    sag_pred = axs[1, 2].imshow(slice, cmap='tab20', vmin=0, vmax=60)
    axs[1, 2].set_title('Sagittal Prediction')
    axs[1, 2].axis('off')
   
    # Coronal (XZ plane at height H)
    slice = t1_voxels[:, coronal_idx, :]
    cor_img = axs[2, 0].imshow(slice, cmap='gray')
    axs[2, 0].set_title(f'Coronal Input Slice {coronal_idx}')
    axs[2, 0].axis('off')

    cor_lbl = axs[2, 1].imshow(labels[:, coronal_idx, :], cmap='tab20', vmin=0, vmax=60)
    axs[2, 1].set_title('Coronal Ground Truth')
    axs[2, 1].axis('off')

    slice = np.squeeze(preds, axis=1)
    slice = slice[:, coronal_idx, :]
    cor_pred = axs[2, 2].imshow(slice, cmap='tab20', vmin=0, vmax=60)
    axs[2, 2].set_title('Coronal Prediction')
    axs[2, 2].axis('off')

    # Sliders
    ax_slider_axial = plt.axes([0.25, 0.15, 0.5, 0.02])
    slider_axial = Slider(ax_slider_axial, 'Axial', 0, target_size - 1, valinit=axial_idx, valfmt='%0.0f')

    ax_slider_sag = plt.axes([0.25, 0.1, 0.5, 0.02])
    slider_sag = Slider(ax_slider_sag, 'Sagittal', 0, target_size - 1, valinit=sagittal_idx, valfmt='%0.0f')

    ax_slider_cor = plt.axes([0.25, 0.05, 0.5, 0.02])
    slider_cor = Slider(ax_slider_cor, 'Coronal', 0, target_size - 1, valinit=coronal_idx, valfmt='%0.0f')

    def update_axial(val):
        idx = int(slider_axial.val)
        slice = t1_voxels[idx, :, :]
        ax_img.set_data(slice)
        ax_lbl.set_data(labels[idx])
        slice = np.squeeze(preds, axis=1)
        slice = slice[idx, :, :]
        ax_pred.set_data(slice)
        axs[0, 0].set_title(f'Axial Input Slice {idx}')
        fig.canvas.draw_idle()

    def update_sag(val):
        idx = int(slider_sag.val)
        slice = t1_voxels[:, :, idx]
        sag_img.set_data(slice)
        sag_lbl.set_data(labels[:, :, idx])
        slice = np.squeeze(preds, axis=1)
        slice = slice[:, :, idx]
        sag_pred.set_data(slice)
        axs[1, 0].set_title(f'Sagittal Input Slice {idx}')
        fig.canvas.draw_idle()

    def update_cor(val):
        idx = int(slider_cor.val)
        slice = t1_voxels[:, idx, :]
        cor_img.set_data(slice)
        cor_lbl.set_data(labels[:, idx, :])
        slice = np.squeeze(preds, axis=1)
        slice = slice[:, idx, :]
        cor_pred.set_data(slice)
        axs[2, 0].set_title(f'Coronal Input Slice {idx}')
        fig.canvas.draw_idle()

    slider_axial.on_changed(update_axial)
    slider_sag.on_changed(update_sag)
    slider_cor.on_changed(update_cor)

    plt.show()

def train_epoch(model, loader, criterion, optimizer, device=None, grad_accum_steps=1):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, (voxels, labels) in enumerate(loader):
        voxels, labels = voxels.to(device), labels.to(device)
        outputs = model(pixel_values=voxels).logits  # (B, C, H, W)
        loss = criterion(outputs, labels) / grad_accum_steps
        loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps

    if (step + 1) % grad_accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / len(loader)

def test_epoch(model, loader, criterion, device=None):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for voxels, labels in loader:
            voxels, labels = voxels.to(device), labels.to(device)
            outputs = model(pixel_values=voxels).logits
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    return total_loss / len(loader)

# for testing
def dice_score(pred, target, epsilon=1e-6):
    pred = torch.softmax(pred, dim=1)
    # [B,H,W]
    target_one_hot = F.one_hot(target.long(), num_classes=pred.shape[1])
    target_one_hot = target_one_hot.permute(0, 3, 1, 2).float() # [B,C,H,W]
    intersection = (pred * target_one_hot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
    dice = (2. * intersection + epsilon) / (union + epsilon)
    return dice.mean()

ce_loss = nn.CrossEntropyLoss(ignore_index=0)
def hybrid_loss(pred, target):
    return ce_loss(pred, target) + inverse_volume_dice_loss(pred, target)

def inverse_volume_dice_loss(pred, target, epsilon=1e-6):
    pred = F.softmax(pred, dim=1)  # (B, C, H, W)
    # One-hot encode target (B, H, W, C) 
    target_one_hot = F.one_hot(target.long(), num_classes=pred.shape[1]) # 
    # (B, C, H, W)
    target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()

    # Compute per-class intersection and union
    intersection = (pred * target_one_hot).sum(dim=(2, 3))  # (B, C)
    pred_sum     = pred.sum(dim=(2, 3))                     # (B, C)
    target_sum   = target_one_hot.sum(dim=(2, 3))           # (B, C), Count how many pixels belong to each class
    union        = pred_sum + target_sum                    # (B, C)
    dice_per_class = (2 * intersection + epsilon) / (union + epsilon)

    # Inverse-volume weights based on ground-truth region sizes
    mask = (target_sum > 0).float()  # (B, C), 1 if class exists, e.g., [1, 1, 0]
    weights = mask / (target_sum + epsilon)  # (B, C), Assign more importance to rare (small) classes
    weights[:, 0] = weights[:, 0] / 3 # make bkgnd contribution smaller
    # Normalize per sample
    weights = weights / weights.sum(dim=1, keepdim=True)
    # weights[:, 0] = 0  # remove background contribution

    # Weighted average dice loss per sample, then per batch
    dice_loss = (1 - dice_per_class) * weights # (B, C)
    return dice_loss.sum(dim=1).mean()         # scalar

if __name__ == "__main__":
    '''
    config = SegformerConfig(
        num_labels=61,
        image_size=64,                   
        num_channels=total_chs,
        patch_sizes=[9, 5, 5, 5], # @TODO try diff patch sizes
        strides=[1, 2, 2, 2], # Total downsample = 8
        hidden_sizes=[16, 32, 80, 80],
        depths=[2, 3, 3, 2],
        mlp_ratios=[2, 2, 2, 2],          
        decoder_hidden_size=64,
        reshape_last_stage=True,
    )
    '''
    '''
    config = SegformerConfig(
        num_labels=61,
        image_size=192,                   
        num_channels=total_chs,
        patch_sizes=[3, 3, 5, 7],
        strides=[1, 1, 2, 2], # Total downsample = 4
        hidden_sizes=[32, 64, 160, 256],
        depths=[3, 4, 4, 3],
        mlp_ratios=[4, 4, 4, 4],          
        decoder_hidden_size=512,
        reshape_last_stage=True,
    )
    '''
    #'''
    config = SegformerConfig(
        num_labels=61,
        image_size=target_size,
        num_channels=total_chs,
        patch_sizes=[7, 3, 3, 3],
        strides=[1, 2, 2, 2],             # Total downsample = 8
        hidden_sizes=[32, 64, 160, 256],  # try [64, 128, 320, 512],
        depths=[2, 2, 2, 2],              # try [3, 4, 6, 3]
        mlp_ratios=[4, 4, 4, 4],
        decoder_hidden_size=256,          # try 256-768
        reshape_last_stage=True,
    )
    #'''
    
    model = SegformerForSemanticSegmentation(config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    #torch.manual_seed(42)
    train_data = MRIDataset(split="train", deg=start_deg)
    test_data = MRIDataset(split="test")
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=6) # 8 * 4
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False, num_workers=6) # 16

    train_crit = inverse_volume_dice_loss # hybrid_loss
    test_crit = dice_score
    LR = 1e-5
    # Common variant of Adam for transformers used in segformer paper
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=1/3,
        patience=20,
        threshold=1e-4,          # minimum change to count as “improvement”
        threshold_mode='rel',    # relative change
        cooldown=0,              # epochs to wait after LR change
        min_lr=1e-7              # floor on the LR
    )

    num_epochs = 000
    best_dice = float("-inf")
    save_path = "recreate_1ch_180deg.pth" # "recreate_5ch.pth"
    load = input(f'load {save_path}? [y/n] ')
    load = 'y' in load
    if load:
        try:
            state_dict = torch.load(save_path)
            model.load_state_dict(state_dict["model_state_dict"])
            best_dice = state_dict["best_dice"]
            optimizer.load_state_dict(state_dict["optimizer_state_dict"])
            # update the LR for each param group
            for param_group in optimizer.param_groups:
                param_group['lr'] = LR
        except: # continues if we fail
            print("ERROR loading previous model!") 

    dice_score = test_epoch(model, test_loader, test_crit, device)
    print(f"Dice Score: {dice_score:.4f}")

    for epoch in range(num_epochs):
        # slowly ramp down the amount of data-aug
        curr_deg = start_deg - epoch
        if curr_deg < 0:
            curr_deg = 0
        train_loader.dataset.deg = curr_deg
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss = train_epoch(model, train_loader, train_crit, optimizer, device, grad_accum_steps=4)
        dice_score = test_epoch(model, test_loader, test_crit, device)
        scheduler.step(dice_score)
        print(f"Train Loss: {train_loss:.4f}, Dice Score: {dice_score:.4f}")

        if dice_score > best_dice:
            best_dice = dice_score
            state_dict = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
            }
            torch.save(state_dict, save_path)
            print(f"Saved Best Model (Epoch {epoch+1})")

    viz_data = MRIDataset(split="test", viz=True)
    viz_loader = DataLoader(viz_data, batch_size=1, shuffle=False, num_workers=3)
    visualize_prediction(model, viz_loader, device)