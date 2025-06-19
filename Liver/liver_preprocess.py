import os
from glob import glob
import torch
import shutil
from tqdm import tqdm
import numpy as np
import nibabel
from monai.transforms import (
    Compose,
    LoadImaged,
    ToTensord,
    EnsureChannelFirstD,
    Resized,
    ToTensord,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    ResizeWithPadOrCropd
)
from monai.data import Dataset, DataLoader, CacheDataset
from monai.utils import set_determinism
from monai.utils import first
import matplotlib.pyplot as plt

def prepare(in_dir, pixdim=(1.5, 1.5, 2.0), a_min=-200, a_max=200, spatial_size=(256, 256, 128), cache=False):
    set_determinism(seed=0)
    Train_images = sorted(glob(os.path.join(in_dir, 'TrainVolumes', '*.nii.gz')))
    Train_labels = sorted(glob(os.path.join(in_dir, 'TrainSegmentation', '*.nii.gz')))
    val_images = sorted(glob(os.path.join(in_dir, 'TestVolumes', '*.nii.gz')))
    val_labels = sorted(glob(os.path.join(in_dir, 'TestSegmentation', '*.nii.gz')))
    Train_files = [{"image": image_name, "label": label_name} for image_name, label_name in zip(Train_images, Train_labels)]
    Val_files = [{"image": image_name, "label": label_name} for image_name, label_name in zip(val_images, val_labels)]
    print(Train_files)
    print(Val_files)
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstD(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=pixdim),
        ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=spatial_size),
        ToTensord(keys=["image", "label"])
    ])
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstD(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=pixdim),
        ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="label"),
        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=spatial_size),
        ToTensord(keys=["image", "label"])
    ])
    if cache:
        train_ds = CacheDataset(data=Train_files, transform=train_transforms, cache_rate=1.0)
        train_loader = DataLoader(train_ds, batch_size=1)
        test_ds = CacheDataset(data=Val_files, transform=val_transforms, cache_rate=1.0)
        test_loader = DataLoader(test_ds, batch_size=1)
        return train_loader, test_loader
    else:
        train_ds = Dataset(data=Train_files, transform=train_transforms)
        train_loader = DataLoader(train_ds, batch_size=1)
        test_ds = Dataset(data=Val_files, transform=val_transforms)
        test_loader = DataLoader(test_ds, batch_size=1)
        return train_loader, test_loader
