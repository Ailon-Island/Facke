import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

class IDExtract(nn.Module):
    def __init__(self):
        super(IDExtract, self).__init__()

        self.INnorm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # normalization of ImageNet

    def forward(self, netArc, img):
        img_interp = F.interpolate(img, size=(112,112))
        img_interp = self.INNorm(img_interp)
        latent_ID = netArc(img_interp)

        return latent_ID