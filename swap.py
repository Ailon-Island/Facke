import os
import re
import shutil
import numpy as np
import math
from collections import OrderedDict
import time
import tqdm
import warnings
import matplotlib.pyplot as plt
from PIL import Image

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
ImageNet_mean, ImageNet_std, Common_mean, Common_std = torch.Tensor(ImageNet_mean), torch.Tensor(
    ImageNet_std), torch.Tensor(Common_mean), torch.Tensor(ImageNet_mean)


def logger(msg):
    save_path = os.path.join(opt.output_path, opt.name)
    file_name = os.path.join(save_path, 'swap_log.txt')

    with open(file_name, 'a') as log_file:
        log_file.write("{}\n".format(msg))
    print(msg)


def read_img(pth, transform, num=8):
    img = []
    for i in range(num):
        img += [Image.open(os.path.join(pth, '{}.jpg'.format(i))).convert('RGB')]
        img[-1] = transform(img[-1])

    img = torch.stack(img)

    return img

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    # args = create_argparser().parse_args()
    opt = TestOptions().parse()

    save_path = os.path.join(opt.output_path, opt.name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(os.path.join(save_path, 'swap_opt.txt')):
        file_name = os.path.join(save_path, 'swap_opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(vars(opt).items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')

    file_name = os.path.join(save_path, 'swap_log.txt')
    with open(file_name, 'a') as log_file:
        log_file.write('===============swapping start=================\n')

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





    print("swapping faces...")
    count = 0
    model.eval()

    ArcExtract = IDExtractor(opt, model='Arcface')
    CosExtract = IDExtractor(opt, model='Cosface')

    IDloss = IDLoss()
    Recloss = nn.L1Loss()

    if opt.model == 'ILVR':
        def swap(img_source, img_target, latent_ID):
            return denorm(model.swap(img_source, img_target))
    elif opt.model == 'CVAE':  # CVAE and CVAE-GAN
        def swap(img_source, img_target, latent_ID):
            return model(img_target, latent_ID)
    else:
        def swap(img_source, img_target, latent_ID):
            return model(img_source, img_target, latent_ID, latent_ID)  # latent_ID_target is not used in eval()

    model.load(opt.epoch_label)

    img_source = read_img(opt.source_path, transform, opt.num_to_swap)
    img_target = read_img(opt.target_path, transform, opt.num_to_swap)

    with torch.no_grad():
        img_source, img_target = img_source.to(device), img_target.to(device)
        latent_ID = ArcExtract(img_source)

        imgs = []
        zero_img = (torch.zeros_like(img_source[0, ...]))
        imgs.append(zero_img.cpu().numpy())
        save_img_source = (denorm(img_source.cpu())).numpy()
        save_img_target = (denorm(img_target.cpu())).numpy()

        for r in range(opt.num_to_swap):
            imgs.append(save_img_source[r, ...])

        for i in range(opt.num_to_swap):
            imgs.append(save_img_target[i, ...])

            image_infer = img_target[i, ...].repeat(opt.num_to_swap, 1, 1, 1)
            img_fake = swap(img_source, image_infer, latent_ID).cpu().numpy()

            for j in range(opt.num_to_swap):
                imgs.append(img_fake[j, ...])

    logger('Face swapped successfully.')

    imgs = np.stack(imgs, axis=0).transpose(0, 2, 3, 1)
    plot_batch(imgs, os.path.join(save_path, '{}_{}.jpg'.format(opt.name, opt.swap_title)))

    file_name = os.path.join(save_path, 'swap_log.txt')
    with open(file_name, 'a') as log_file:
        log_file.write('===============swapping end=================\n')