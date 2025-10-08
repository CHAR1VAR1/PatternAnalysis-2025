import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

class ADNIDataset(Dataset):
    """
    A PyTorch Dataset for loading 2D MRI image slices from the ADNI dataset,
    organized into Alzheimer's disease (AD) and normal control (NC) categories.

    Args:
        root_dir (str): Path to the AD_NC directory containing "train" and "test".
        split (str): Which dataset split to use, "train" or "test".
        transform (callable, optional): Optional transform to be applied on an image.

    Attributes:
        samples (list): List of tuples (image_path, label) for all samples
                        in the specified split.
    """

    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = os.path.join(root_dir, split)   # path automatically goes to train folder
        self.transform = transform
        self.samples = []   # (image_path, label)

        # get all (image_path, label) pairs
        for label_name, label in [("AD", 1), ("NC", 0)]:
            class_dir = os.path.join(self.root_dir, label_name)
            for fname in os.listdir(class_dir):
                self.samples.append((os.path.join(class_dir, fname), label))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

def get_dataloaders(root_dir, batch_size=16):
    """
    Create PyTorch DataLoaders for the ADNI dataset.

    This function initializes ADNIDataset instances for the training and
    testing splits, applies preprocessing transforms (resize, tensor conversion,
    normalisation), and returns DataLoaders for batched access.

    Args:
        root_dir (str): Path to the AD_NC directory containing 'train' and 'test'.
        batch_size (int, optional): Number of samples per batch. Default is 16.

    Returns:
        tuple:
            - train_loader (DataLoader): DataLoader for the training set,
                                         with shuffling enabled.
            - test_loader (DataLoader): DataLoader for the test set,
                                        with shuffling disabled.
    """
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1159], std=[0.2199])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1159], std=[0.2199])
    ])

    train_dataset = ADNIDataset(os.path.join(root_dir, "AD_NC"), split="train", transform=train_transform)
    test_dataset = ADNIDataset(os.path.join(root_dir, "AD_NC"), split="test", transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader