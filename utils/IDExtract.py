import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

class IDExtractor(nn.Module):
    def __init__(self, opt, model='Arcface'):
        super(IDExtractor, self).__init__()

        if model == 'Arcface':
            checkpoint = opt.Arc_path
        elif model == 'CosFace':
            netCos_checkpoint = opt.Cos_path

        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint)
        else:
            checkpoint = torch.load(checkpoint,map_location=torch.device('cpu'))
        self.net = checkpoint['model'].module
        if torch.cuda.is_available():
            self.net = self.net.to('cuda')
        else:
            self.net = self.net.to('cpu')
        self.net.eval()

        # transformation
        self.INnorm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # normalization of ImageNet
        self.transform = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))\
            if model == 'CosFace' else self.INnorm

    def forward(self, img):
        img_interp = F.interpolate(img, size=(112,112))
        # img_interp = self.transform(img_interp)
        latent_ID = self.net(img_interp)
        latent_ID = F.normalize(latent_ID, p=2, dim=1)

        return latent_ID