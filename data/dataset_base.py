import os
from torch.utils.data import Dataset
from torchvision import datasets

class DatasetBase(Dataset):
    def __init__(self, data_dir, transform=None, target_transform=None):
        self.img_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform


    def __len__(self):
        pass


    def __getitem__(self, idx):
        pass