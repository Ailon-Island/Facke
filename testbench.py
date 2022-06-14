import os
import shutil
import numpy as np
import math
from collections import OrderedDict
import time
import tqdm
import warnings

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from options.test_options import TestOptions
from models.models import create_model
from data.VGGface2HQ import VGGFace2HQDataset
from utils.visualizer import Visualizer
from utils import utils
from utils.loss import get_loss_dict
from utils.plot import plot_batch
from utils.loss import IDLoss
from utils.IDExtract import IDExtractor



ImageNet_mean = [0.485, 0.456, 0.406]
ImageNet_std = [0.229, 0.224, 0.225]
Common_mean = [0.5, 0.5, 0.5]
Common_std = [0.5, 0.5, 0.5]

if __name__ == '__main__':
    # args = create_argparser().parse_args()
    opt = TestOptions().parse()

    # th.manual_seed(0)

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    print("Initiating model...")
    model = create_model(opt)
    print("Model initiated.")

    # data transform
    if opt.model == 'ILVR':
        transform_mean, transform_std = Common_mean, Common_std
    else:
        transform_mean, transform_std = ImageNet_mean, ImageNet_std

    norm = transforms.Normalize(transform_mean, transform_std)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((opt.image_size, opt.image_size)),
        norm
    ])
    denorm = transforms.Compose([
        transforms.Normalize([0, 0, 0], 1 / transform_std),
        transforms.Normalize(-transform_mean, [1, 1, 1])
    ])

    print("Generating data loaders...")
    test_data = VGGFace2HQDataset(opt, isTrain=False, transform=transform, is_same_ID=False, auto_same_ID=False)
    test_loader = DataLoader(dataset=test_data, batch_size=opt.batchSize, shuffle=True, num_workers=opt.nThreads,
                             worker_init_fn=test_data.set_worker)
    print("Dataloaders ready.")

    print("testing models...")
    count = 0
    model.eval()

    ArcExtract = IDExtractor(opt, model='Arcface')
    CosExtract = IDExtractor(opt, model='Cosface')

    IDloss = IDLoss()

    if opt.model == 'ILVR':
        def swap(img_source, img_target, latent_ID):
            model.swap(img_source, img_target)
    else:
        def swap(img_source, img_target, latent_ID):
            model(img_source, img_target, latent_ID, latent_ID_target) # latent_ID_target is not used in eval()

    metrics = {}
    metrics['ID Loss', 'ID Retrieval', 'Recon Loss'] = 0, 0, 0

    for (img_source, img_target), (latent_ID, _), is_same_ID in tqdm.tqdm(test_loader):
        img_source, img_target, latent_ID = img_source.to(device), img_target.to(device), latent_ID.to(device)

        img_fake = swap(img_source, img_target, latent_ID)
        img_fake = norm(img_fake)

        fake_ID = ArcExtract(img_fake)
        metrics['ID Loss'] += IDloss(fake_ID, latent_ID).detach().item()

        latent_ID = CosExtract(img_source)
        fake_ID = CosExtract(img_fake)

