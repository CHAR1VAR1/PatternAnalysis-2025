import os
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
import torchvision.transforms as transforms

class ADNIDataset(Dataset):
    """
    PyTorch Dataset class for loading and preprocessing ADNI brain MRI scans.

    This dataset takes a list of file paths (NIfTI format) and corresponding labels,
    loads each scan, extracts a representative 2D slice, normalizes the image, and 
    optionally applies transforms before returning a tensor with its label.

    Args:
        file_paths (list of str): List of paths to NIfTI image files (.nii or .nii.gz).
        labels (list of int): List of integer labels corresponding to each file.
                              (e.g., 0 = Normal, 1 = Alzheimer's).
        transform (callable, optional): A torchvision transform or custom function
                                        applied to the image tensor.

    Returns:
        tuple: (image_tensor, label), where
            - image_tensor (torch.FloatTensor): Preprocessed 2D brain slice, shape [1, H, W].
            - label (torch.LongTensor): Class label for the image.
    """

def get_dataloaders(train_files, val_files, test_files,
                    train_labels, val_labels, test_labels,
                    batch_size=16):
    """
    Creates PyTorch DataLoader objects for training, validation, and testing.

    Given lists of file paths and labels for each split, this function constructs
    ADNIDataset objects, applies preprocessing/transforms, and wraps them in
    DataLoader objects for efficient batching and iteration.

    Args:
        train_files (list of str): File paths for training images.
        val_files (list of str): File paths for validation images.
        test_files (list of str): File paths for testing images.
        train_labels (list of int): Class labels for training images.
        val_labels (list of int): Class labels for validation images.
        test_labels (list of int): Class labels for testing images.
        batch_size (int, optional): Number of samples per batch. Defaults to 16.

    Returns:
        tuple: (train_loader, val_loader, test_loader), where each is a
               torch.utils.data.DataLoader ready for use in training loops.
    """