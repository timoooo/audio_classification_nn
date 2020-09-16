import numpy as np
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
import torch


def get_train_and_validation_data_loader(data_path="images", validation_split_ratio=0.1, seed=42, size=None):
    if size is None:
        size = [160, 120]
    data = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=transforms.Compose([
            transforms.Resize((size[0], size[1])),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )
    dataset_size = len(data)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split_ratio * dataset_size))
    np.random.seed(seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(
        data,
        num_workers=4,
        batch_size=1,
        sampler=train_sampler
    )
    valid_loader = DataLoader(
        data,
        num_workers=4,
        batch_size=1,
        sampler=valid_sampler
    )
    return train_loader, valid_loader
