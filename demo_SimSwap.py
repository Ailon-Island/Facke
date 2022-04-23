import cv2
import numpy as np
from options.test_options import TestOptions
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
from utils import utils

warnings.filterwarnings("ignore")


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.name = 'SimSwap_WO_intra-ID_random'

    transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((opt.image_size, opt.image_size)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    detransformer_Arcface = transforms.Compose([
        transforms.Normalize([0, 0, 0], [1 / 0.229, 1 / 0.224, 1 / 0.225]),
        transforms.Normalize([-0.485, -0.456, -0.406], [1, 1, 1])
    ])

    if opt.fp16:
        from torch.cuda.amp import autocast

    print("Generating data loaders...")
    test_data = VGGFace2HQDataset(opt, isTrain=False,  transform=transformer_Arcface, is_same_ID=True, auto_same_ID=False)
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=True, num_workers=1)
    print("Datasets ready.")

    torch.nn.Module.dump_patches = True

    model = create_model(opt)
    model.eval()
    with torch.no_grad():
        it = iter(test_loader)
        (img_source, img_target), (latent_ID, latent_ID_target), _ = next(it)
        img_source, img_target, latent_ID, latent_ID_target = img_source.to('cuda'), img_target.to('cuda'), latent_ID.to('cuda'), latent_ID_target.to('cuda')
        img_fake = model(img_source, img_target, latent_ID, latent_ID_target)

        # img_source = detransformer_Arcface(img_source)
        # img_target = detransformer_Arcface(img_target)
        # img_fake = detransformer_Arcface(img_target)

        img_source = utils.tensor2im(img_source[0])
        img_target = utils.tensor2im(img_target[0])
        img_fake = utils.tensor2im(img_fake.data[0])

        plt.figure(1)
        plt.imshow(img_source)
        utils.save_image(img_source, opt.output_path + 'source.jpg')

        plt.figure(2)
        plt.imshow(img_target)
        utils.save_image(img_target, opt.output_path + 'target.jpg')

        plt.figure(3)
        plt.imshow(img_fake)
        utils.save_image(img_fake, opt.output_path + 'result.jpg')

        print("done!")

        plt.show()
