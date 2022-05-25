###############################################################################
# Code adapted and modified from
# https://github.com/jychoi118/ilvr_adm
###############################################################################
import os

import tqdm
from options.test_options import TestOptions

from data.VGGface2HQ import VGGFace2HQDataset
from utils.guided_diffusion import  logger

import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from models.models import create_model

from utils.plot import plot_batch



class Transform:
    def __int__(self):
        super(Transform, self).__int__()

    def __call__(self, x):
        return x * 2 - 1



class DeTransform:
    def __int__(self):
        super(DeTransform, self).__int__()

    def __call__(self, x):
        return (x + 1) / 2



if __name__ == '__main__':
    # args = create_argparser().parse_args()
    opt = TestOptions().parse()
    opt.model = 'ILVR'

    # th.manual_seed(0)

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    logger.configure(dir=opt.output_path)

    # logger.log("loading data...")
    # # data = load_reference(
    # #     opt.base_samples,
    # #     args.batch_size,
    # #     image_size=args.image_size,
    # #     class_cond=args.class_cond,
    # # )

    logger.log("Initiating model...")
    model = create_model(opt)
    logger.log("Model initiated.")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((opt.image_size, opt.image_size)),
        Transform()
    ])
    detransform = DeTransform()

    logger.log("Generating data loaders...")
    test_data = VGGFace2HQDataset(opt, isTrain=False, transform=transform, is_same_ID=True, auto_same_ID=True)
    test_loader = DataLoader(dataset=test_data, batch_size=opt.batchSize, shuffle=True, num_workers=opt.nThreads,
                             worker_init_fn=test_data.set_worker)
    logger.log("Dataloaders ready.")

    logger.log("creating samples...")
    count = 0
    for (img_source, _), _, is_same_ID in tqdm.tqdm(test_loader):
        if count >= opt.ntest:
            break
        count += img_source.shape[0]

        img_source = img_source.to(device)

        # display images
        sample_size = min(8, opt.batchSize)

        output_pth = os.path.join(opt.output_path, opt.name)
        if not os.path.exists(output_pth):
            os.mkdir(output_pth)
        sample_path = os.path.join(output_pth, 'samples')
        if not os.path.exists(sample_path):
            os.mkdir(sample_path)

        with torch.no_grad():
            img_source = img_source[:sample_size]

            imgs = []
            zero_img = (torch.zeros_like(img_source[0, ...]))
            imgs.append(zero_img.cpu().numpy())
            save_img = (detransform(img_source.cpu())).numpy()

            for r in range(sample_size):
                imgs.append(save_img[r, ...])

            for i in range(sample_size):
                imgs.append(save_img[i, ...])

                image_infer = img_source[i, ...].repeat(sample_size, 1, 1, 1)
                img_fake = model.swap(img_source, image_infer)
                img_fake = (detransform(img_fake.cpu())).numpy()

                for j in range(sample_size):
                    imgs.append(img_fake[j, ...])

            imgs = np.stack(imgs, axis=0).transpose(0, 2, 3, 1)
            plot_batch(imgs, os.path.join(sample_path, 'sample_' + str(count) + '.jpg'))


        logger.log(f"created {count} samples")
    logger.log("sampling complete")

