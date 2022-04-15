import numpy as np
from options.train_options import TrainOptions
import torch
from torch import nn
from utils.IDExtract import IDExtractor
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from models.models import create_model
from data.VGGface2HQ import VGGFace2HQDataset, ComposedLoader
import time
import matplotlib.pyplot as plt
import warnings
from utils.loss import IDLoss

warnings.filterwarnings("ignore")
torch.autograd.set_detect_anomaly(True)



if __name__ == '__main__':
    opt = TrainOptions().parse()

    transformer_to_tensor = transforms.ToTensor()


    if opt.fp16:
        from torch.cuda.amp import autocast

    print("Generating data loaders...")
    train_data = VGGFace2HQDataset(opt, isTrain=True, transform=transformer_to_tensor, is_same_ID=True, auto_same_ID=False, random_in_ID=False, force_new_latents=True)
    train_loader = DataLoader(dataset=train_data, batch_size=opt.batchSize, shuffle=False, num_workers=3)
    test_data = VGGFace2HQDataset(opt, isTrain=False, transform=transformer_to_tensor, is_same_ID=True, auto_same_ID=False, random_in_ID=False, force_new_latents=True)
    test_loader = DataLoader(dataset=test_data, batch_size=opt.batchSize, shuffle=False, num_workers=3)
    print("Dataloaders ready.")

    torch.nn.Module.dump_patches = True

    train_size = len(train_data)
    print('Creating latents for train set...')
    cnt = 0
    for _ in enumerate(train_loader, start=1):
        cnt += opt.batchSize
        print('Generated {}/{}({:.3%})'.format(cnt, train_size, 1. * cnt / train_size))
    print('Train set successfully constructed.')

    test_size = len(test_data)
    print('Creating latents for test...')
    cnt = 0
    for _ in enumerate(test_loader, start=1):
        cnt += opt.batchSize
        print('Generated {}/{}({.3%})'.format(cnt, test_size, 1. * cnt / test_size))
    print('Test set successfully constructed.')


