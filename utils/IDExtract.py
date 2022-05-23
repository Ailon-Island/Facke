import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

class IDExtractor(nn.Module):
    def __init__(self, opt):
        super(IDExtractor, self).__init__()

        netArc_checkpoint = opt.Arc_path
        if torch.cuda.is_available():
            netArc_checkpoint = torch.load(netArc_checkpoint)
        else:
            netArc_checkpoint = torch.load(netArc_checkpoint,map_location=torch.device('cpu'))
        self.netArc = netArc_checkpoint['model'].module
        if torch.cuda.is_available():
            self.netArc = self.netArc.to('cuda')
        else:
            self.netArc = self.netArc.to('cpu')
        self.netArc.eval()
        self.INnorm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # normalization of ImageNet

    def forward(self, img):
        img_interp = F.interpolate(img, size=(112,112))
        img_interp = self.INnorm(img_interp)
        latent_ID = self.netArc(img_interp)
        latent_ID = F.normalize(latent_ID, p=2, dim=1)

        return latent_ID