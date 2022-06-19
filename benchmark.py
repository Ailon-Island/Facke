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
ImageNet_mean, ImageNet_std, Common_mean, Common_std = torch.Tensor(ImageNet_mean), torch.Tensor(ImageNet_std), torch.Tensor(Common_mean), torch.Tensor(ImageNet_mean)


def logger(msg):
    save_path = os.path.join(opt.checkpoints_dir, opt.name)
    file_name = os.path.join(save_path, 'benchmark_log.txt')

    with open(file_name, 'a') as log_file:
        log_file.write("{}\n".format(msg))
    print(msg)



if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    # args = create_argparser().parse_args()
    opt = TestOptions().parse()

    save_path = os.path.join(opt.checkpoints_dir, opt.name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(os.path.join(save_path, 'benchmark_opt.txt')):
        file_name = os.path.join(save_path, 'benchmark_opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(vars(opt).items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
            
    file_name = os.path.join(save_path, 'benchmark_log.txt')
    with open(file_name, 'a') as log_file:
        log_file.write('===============benchmark start=================\n')

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
    Recloss = nn.L1Loss()

    if opt.model == 'ILVR':
        def swap(img_source, img_target, latent_ID):
            return model.swap(img_source, img_target)
    elif opt.model == 'CVAE': # CVAE and CVAE-GAN
        def swap(img_source, img_target, latent_ID):
            return model(img_target, latent_ID)
    else:
        def swap(img_source, img_target, latent_ID):
            return model(img_source, img_target, latent_ID, latent_ID) # latent_ID_target is not used in eval()

    if opt.model == 'ILVR':
        dict_dir = 'DDPM'
    elif opt.model == 'SimSwap':
        dict_dir = 'D1'
    else:
        dict_dir = 'D'

    if opt.name == 'ILVR_pretrained':
        check_list = ['pretrained']
    else:
        check_list = os.path.join(opt.checkpoints_dir, opt.name, dict_dir)
        check_list = os.listdir(check_list)
        check_list = [i.split('.')[0] for i in check_list]
        check_list.remove('latest')
        check_list.sort(key=lambda x: int(re.findall('\d+', x)[0]))
        check_list = [check_list[0]] + check_list[-1::-opt.benchmark_skip][::-1]
        check_list = check_list[1:] if check_list[0] == check_list[1] else check_list

    metrics = {'ID Loss': [], 'ID Retrieval': [], 'Recon Loss': []}
    best = {'ID Retrieval': (None, np.inf), 'Recon Loss': (None, np.inf), 'ID Retrieval + Recon Loss': (None, np.inf)}


    for epoch_label in check_list:
        model.load(epoch_label)

        for k in metrics.keys():
            metrics[k] += [0.]
        count = 0

        with torch.no_grad():
            for (img_source, img_target), (latent_ID, _), _ in tqdm.tqdm(test_loader):
                img_source, img_target, latent_ID = img_source.to(device), img_target.to(device), latent_ID.to(device)

                img_fake = swap(img_source, img_target, latent_ID)
                img_fake = norm(img_fake)

                fake_ID = ArcExtract(img_fake)
                metrics['ID Loss'][-1] += IDloss(fake_ID, latent_ID).detach().item()

                latent_ID = CosExtract(img_source)
                fake_ID = CosExtract(img_fake)
                metrics['ID Retrieval'][-1] += IDloss(fake_ID, latent_ID).detach().item()

                # reconstruction
                img_fake = swap(img_source, img_source, latent_ID)
                img_fake = norm(img_fake)

                metrics['Recon Loss'][-1] += Recloss(img_source, img_fake)

                count += img_source.shape[0]
                if count >= opt.benchmark_coarse:
                    break

        # calculate mean
        for k in metrics:
            metrics[k][-1] /= count
        logger("[iter {}] ID Loss: {:.3f}, ID Retrieval: {:.3f}, Recon Loss: {:.3f}.".format(epoch_label, metrics['ID Loss'][-1], metrics['ID Retrieval'][-1], metrics['Recon Loss'][-1]))

        # check if is best
        if metrics['ID Retrieval'][-1] < best['ID Retrieval'][1]:
            best['ID Retrieval'] = (epoch_label, metrics['ID Retrieval'][-1])
        if metrics['Recon Loss'][-1] < best['Recon Loss'][1]:
            best['Recon Loss'] = (epoch_label, metrics['Recon Loss'][-1])
        if metrics['ID Retrieval'][-1] + metrics['Recon Loss'][-1] < best['ID Retrieval + Recon Loss'][1]:
            best['ID Retrieval + Recon Loss'] = (epoch_label, metrics['ID Retrieval'][-1] + metrics['Recon Loss'][-1])

    for (metric, (epoch_label, _)) in best.items():
        model.load(epoch_label)

        metrics_tmp = {'ID Loss': 0., 'ID Retrieval': 0., 'Recon Loss': 0.}
        count = 0

        with torch.no_grad():
            for (img_source, img_target), (latent_ID, _), _ in tqdm.tqdm(test_loader):
                img_source, img_target, latent_ID = img_source.to(device), img_target.to(device), latent_ID.to(device)

                img_fake = swap(img_source, img_target, latent_ID)
                img_fake = norm(img_fake)

                fake_ID = ArcExtract(img_fake)
                metrics_tmp['ID Loss'] += IDloss(fake_ID, latent_ID).detach().item()

                latent_ID = CosExtract(img_source)
                fake_ID = CosExtract(img_fake)
                metrics_tmp['ID Retrieval'] += IDloss(fake_ID, latent_ID).detach().item()

                # reconstruction
                img_fake = swap(img_source, img_source, latent_ID)
                img_fake = norm(img_fake)

                metrics_tmp['Recon Loss'] += Recloss(img_source, img_fake)

                count += img_source.shape[0]
                if count >= opt.benchmark_fine:
                    break

        # calculate mean
        for k in metrics_tmp:
            metrics_tmp[k] /= count

        best[metric] = (epoch_label, metrics_tmp)


    print(best)
    logger("Best ID Retrieval:\t [iter {}] ID Loss: {:.3f}, ID Retrieval: {:.3f}, Recon Loss: {:.3f}.".format(best['ID Retrieval'][0], best['ID Retrieval'][1]['ID Loss'], best['ID Retrieval'][1]['ID Retrieval'], best['ID Retrieval'][1]['Recon Loss']))
    logger("Best Recon Loss:\t [iter {}] ID Loss: {:.3f}, ID Retrieval: {:.3f}, Recon Loss: {:.3f}.".format(best['Recon Loss'][0], best['Recon Loss'][1]['ID Loss'], best['Recon Loss'][1]['ID Retrieval'], best['Recon Loss'][1]['Recon Loss']))
    logger("Best ID Retrieval + Recon Loss:\t [iter {}] ID Loss: {:.3f}, ID Retrieval: {:.3f}, Recon Loss: {:.3f}.".format(best['ID Retrieval + Recon Loss'][0], best['ID Retrieval + Recon Loss'][1]['ID Loss'], best['ID Retrieval + Recon Loss'][1]['ID Retrieval'], best['ID Retrieval + Recon Loss'][1]['Recon Loss']))

    benchmark_dir = os.path.join(opt.checkpoints_dir, opt.name, 'benchmark_metrics.pth')
    torch.save(metrics, benchmark_dir)

    file_name = os.path.join(save_path, 'benchmark_log.txt')
    with open(file_name, 'a') as log_file:
        log_file.write('===============benchmark end=================\n')