import  os
import sys
root_path = os.path.join("..", "..")
sys.path.append(root_path)
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from utils import loss
from . import networks
from ..model_base import ModelBase

class GAN(ModelBase):
    def __init__(self):
        super(GAN, self).__init__()


    def init(self, opt):
        ModelBase.init(self, opt)

        self.isTrain = opt.isTrain

        device = torch.device("cuda:0")

        # Generator
        self.G = networks.Generator(in_channels=3, out_channels=3, latent_size=512, num_ID_blocks=9)
        self.G = self.G.to(device)

        # ID network
        netArc_checkpoint = opt.Arc_path
        netArc_checkpoint = torch.load(netArc_checkpoint)
        self.netArc = netArc_checkpoint['model'].module
        self.netArc = self.netArc.to(device)
        self.netArc.eval()

        ### if not training, only Generator needed ###
        if not self.isTrain:
            # load G only
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_net(self.G, opt.epoch_label, 'G', pretrained_path)
            return
        ##############################################

        ### training ###
        self.INnorm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # normalization of ImageNet
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=[1, 1], count_include_pad=False)

        # Discriminators
        if opt.gan_mode == 'original':
            use_sigmoid = True
        else:
            use_sigmoid = False

        self.D1 = networks.Discriminator(in_channels=3, use_sigmoid=use_sigmoid)
        self.D1 = self.D1.to(device)
        self.D2 = networks.Discriminator(in_channels=3, use_sigmoid=use_sigmoid)
        self.D2 = self.D2.to(device)

        # load G and D
        if opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain

            self.load_net(self.G, opt.epoch_label, 'G', pretrained_path)
            self.load_net(self.D1, opt.epoch_label, 'D1', pretrained_path)
            self.load_net(self.D2, opt.epoch_label, 'D2', pretrained_path)

        # loss functions
        self.IDloss = loss.IDLoss()
        self.Recloss = nn.L1Loss()
        self.GANloss = loss.GANLoss(opt.gan_mode, tensor=self.Tensor, opt=opt)
        self.GPloss = loss.GPLoss()
        self.wFMloss = nn.L1Loss()

        # optimizers
        params = list(self.G.parameters())
        self.optim_G = torch.optim.Adam(params, lr=opt.lr, beta=(opt.beta1, 0.999))

        params = list(self.D1.parameters() + self.D2.parameters())
        self.optim_D = torch.optim.Adam(params, lr=opt.lr, beta=(opt.beta1, 0.999))



    def forward(self, x):
        pass