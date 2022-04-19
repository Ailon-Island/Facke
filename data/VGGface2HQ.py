import os
import numpy as np
import torch
import torch.nn.functional as F
from .dataset_base import DatasetBase
from torchvision import datasets, transforms
import random
from utils import utils
from utils.IDExtract import IDExtractor

class VGGFace2HQDataset(DatasetBase):
    def __init__(self, opt, isTrain=True, transform=None, is_same_ID=True, auto_same_ID=True):  #isTrain=True, data_dir='datasets\\VGGface2_HQ', is_same_ID=True, transform=None):
        self.opt = opt
        set = 'train' if isTrain else 'test'
        self.data_dir = os.path.join(opt.dataroot, set)
        img_dir = os.path.join(self.data_dir, 'images')
        self.dataset = datasets.ImageFolder(img_dir)
        self.batch_size = opt.batchSize
        self.transform = transform
        self.is_same_ID = is_same_ID
        self.auto_same_ID = auto_same_ID
        self.intra_ID_random = not opt.no_intra_ID_random
        self.sample_cnt = 0
        self.label_ranges = [len(self.dataset.imgs)] * (len(self.dataset.classes) + 1)
        for i, target in enumerate(self.dataset.targets):
            self.label_ranges[target] = min(self.label_ranges[target], i)
        self.ID_extract = None


    def toggle_is_same_ID(self):
        self.is_same_ID = not self.is_same_ID


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx_source):
        # get source image
        img_source = self.dataset[idx_source][0]

        # get index of target image
        same_outputs = False
        label_source = self.dataset.targets[idx_source]
        if self.is_same_ID:
            # pick target image from the same ID
            if self.intra_ID_random:
                idx_target = random.randint(self.label_ranges[label_source], self.label_ranges[label_source + 1] - 2)
                idx_target = idx_target + 1 if idx_target >= idx_source else idx_target
            else:
                same_outputs = True
        else:
            # pick target image from a different ID
            label_target = random.randint(0, len(self.dataset.classes) - 2)
            label_target = label_target + 1 if label_target >= label_source else label_target
            idx_target = random.randint(self.label_ranges[label_target], self.label_ranges[label_target + 1] - 1)

        # process source image
        latent_id_source = self.get_latent(idx_source)
        if self.transform is not None:
            img_source = self.transform(img_source)

        # get and process target image
        if same_outputs:
            img_target = img_source
            latent_id_target = latent_id_source
        else:
            img_target = self.dataset[idx_target][0]
            latent_id_target = self.get_latent(idx_target)
            if self.transform is not None:
                img_target = self.transform(img_target)

        # toggle the same ID flag
        is_same_ID = self.is_same_ID
        if self.auto_same_ID:
            self.sample_cnt += 1;
            if self.sample_cnt == self.batch_size:
                self.toggle_is_same_ID()
                self.sample_cnt = 0

        return (img_source, img_target), (latent_id_source, latent_id_target), is_same_ID


    def get_latent(self, idx):
        img, label = self.dataset[idx]
        latent_ID_dir = os.path.join(self.data_dir, 'latent-ID')
        utils.mkdirs(latent_ID_dir)
        class_name = self.dataset.classes[label]
        save_class_dir = os.path.join(latent_ID_dir, class_name)
        utils.mkdirs(save_class_dir)
        save_pth = os.path.join(save_class_dir, str(idx)+'.npy')

        if not os.path.exists(save_pth):
            self.generate_latent(img, save_pth)

        latent_ID = np.load(save_pth)
        if (latent_ID.shape != (512,)):
            print(latent_ID.shape)
            self.generate_latent(img, save_pth)
        #latent_ID = latent_ID / np.linalg.norm(latent_ID)
        latent_ID = torch.from_numpy(latent_ID)

        return latent_ID


    def generate_latent(self, img, save_pth):
        if self.ID_extract is None:
            self.ID_extract = IDExtractor(self.opt)
            self.ID_extract.eval()
        with torch.no_grad():
            img = transforms.ToTensor()(img)
            img = img.view(-1, img.shape[0], img.shape[1], img.shape[2])
            img = img.to('cuda')
            latent_ID = self.ID_extract(img).cpu().numpy()
            latent_ID = latent_ID.reshape(-1)
            np.save(save_pth, latent_ID)


# deprecated
class ComposedLoader:
    def __init__(self, loader1, loader2):
        super(ComposedLoader, self).__init__()
        self.loaders = [loader1, loader2]
        self.iters = [iter(loader1), iter(loader2)]
        self.current_iter_id = 1
        self.batch_size = self.loaders[0].batch_size


    def __iter__(self):
        return self


    def __next__(self):
        self.current_iter_id = 1 - self.current_iter_id

        return next(self.iters[self.current_iter_id])


    def __len__(self):
        return len(self.loaders[0]) + len(self.loaders[1])
