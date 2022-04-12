import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

class IDExtractor(nn.Module):
    def __init__(self, netArc):
        super(IDExtractor, self).__init__()

        self.netArc = netArc
        self.INnorm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # normalization of ImageNet

    def forward(self, img):
        img_interp = F.interpolate(img, size=(112,112))
        img_interp = self.INnorm(img_interp)
        latent_ID = self.netArc(img_interp)

        return latent_ID