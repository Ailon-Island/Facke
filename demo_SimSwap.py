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

warnings.filterwarnings("ignore")


if __name__ == '__main__':
    opt = TestOptions().parse()

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
    test_data = VGGFace2HQDataset(isTrain=False, data_dir=opt.dataroot, transform=transformer_Arcface, is_same_ID=False)
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=True, num_workers=8)
    print("Datasets ready.")

    torch.nn.Module.dump_patches = True

    model = create_model(opt)
    model.eval()
    with torch.no_grad():
        it = iter(test_loader)
        (img_source, img_target), _ = next(it)
        img_source, img_target = img_source.detach().to('cuda'), img_target.detach().to('cuda')
        img_fake = model(img_source, img_target)

        img_source = detransformer_Arcface(img_source)
        img_target = detransformer_Arcface(img_target)
        img_fake = detransformer_Arcface(img_target)

        for i in range(img_source.shape[0]):
            if i == 0:
                row1 = img_source[i]
                row2 = img_target[i]
                row3 = img_fake[i]
            else:
                row1 = torch.cat([row1, img_source[i]], dim=2)
                row2 = torch.cat([row2, img_target[i]], dim=2)
                row3 = torch.cat([row3, img_fake[i]], dim=2)

        # full = torch.cat([row1, row2, row3], dim=1).detach()
        full = row1.detach()
        full = full.permute(1, 2, 0)
        output = full.to('cpu')
        output = np.array(output)
        output = output[..., ::-1]

        plt.imshow(output)
        plt.show()
        cv2.imwrite(opt.output_path + 'source.jpg',output)

        full = row2.detach()
        full = full.permute(1, 2, 0)
        output = full.to('cpu')
        output = np.array(output)
        output = output[..., ::-1]

        plt.imshow(output)
        plt.show()
        cv2.imwrite(opt.output_path + 'target.jpg',output)

        full = row3.detach()
        full = full.permute(1, 2, 0)
        output = full.to('cpu')
        output = np.array(output)
        output = output[..., ::-1]

        plt.imshow(output)
        plt.show()
        cv2.imwrite(opt.output_path + 'result.jpg',output)

        print("done!")

