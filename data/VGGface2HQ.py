import os
from .dataset_base import DatasetBase
from torchvision import datasets
import random

class VGGFace2HQDataset(DatasetBase):
    def __init__(self, isTrain=True, data_dir='datasets\\VGGface2_HQ', is_same_ID=True, transform=None):
        set = 'train' if isTrain else 'test'
        self.img_dir = os.path.join(data_dir, set)
        self.transform = transform
        self.dataset = datasets.ImageFolder(self.img_dir)
        self.is_same_ID = is_same_ID
        self.label_ranges = [len(self.dataset.imgs)] * (len(self.dataset.classes) + 1)
        for i, target in enumerate(self.dataset.targets):
            self.label_ranges[target] = min(self.label_ranges[target], i)

    def toggle_is_same_ID(self):
        self.is_same_ID = not self.is_same_ID


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx_source):
        img_source = self.dataset[idx_source][0]

        label_source = self.dataset.targets[idx_source]
        if self.is_same_ID:
            # pick target image from the same ID
            idx_target = random.randint(self.label_ranges[label_source], self.label_ranges[label_source + 1] - 2)
            idx_target = idx_target + 1 if idx_target >= idx_source else idx_target
        else:
            # pick target image from a different ID
            label_target = random.randint(0, len(self.dataset.classes) - 2)
            label_target = label_target + 1 if label_target >= label_source else label_target
            idx_target = random.randint(self.label_ranges[label_target], self.label_ranges[label_target + 1] - 1)

        img_target = self.dataset[idx_target][0]

        if self.transform is not None:
            img_source = self.transform(img_source)
            img_target = self.transform(img_target)

        return (img_source, img_target), self.is_same_ID



class ComposedLoader:
    def __init__(self, loader1, loader2):
        super(ComposedLoader, self).__init__()

        self.iters = [iter(loader1), iter(loader2)]
        self.iter = 0


    def __iter__(self):
        return self


    def __next__(self):
        nex = self.iters[self.iter] = next(self.iters[self.iter])

        self.iter = 1 - self.iter

        return nex
