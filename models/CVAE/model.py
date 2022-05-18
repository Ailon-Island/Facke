import  os
import sys
import time

root_path = os.path.join("..", "..")
sys.path.append(root_path)
import torch
from torch import nn
import torch.nn.functional as F
from utils import loss
from utils.IDExtract import IDExtractor
from . import networks
from ..model_base import ModelBase

class CVAE(ModelBase):
    def __init__(self):
        super(CVAE,self).__init__()
    
    def init(self, opt):
        if opt.verbose:
            print("Initailizing CVAE model...")
        ModelBase.init(self, opt)

        self.isTrain = opt.isTrain
        self.gpu_ids = opt.gpu_ids

        self.iter = 0

        device = torch.device(self.gpu_ids[0])
        #Merge 1

        self.M1 = networks.Merge_Img(in_channel = 3)

        #Encoder
        self.E = networks.Encoder(in_channels= 4)
        self.E = self.E.to(device)

        #Encoder -> MERGE -> DECODER
        self.M2 = networks.Merge_Latent(in_channels=512, out_channels=512, latent_size=512)
        
    
        #Decoder
        self.D = networks.Decoder()
        self.D = self.D.to(device)

    def loss_function(self, recons, input, mu, log_var, weight):
        recons_loss = F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1+ log_var - mu**2 - log_var.exp(),dim=1),dim = 0)
        loss = recons_loss + weight * kld_loss
        return loss

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.rand_like(std)
        return eps* std + mu

    def forward(self, img_source, img_target, latent_ID, latent_ID_target):

        X = self.M1(img_source, img_target)

        mu, log_var = self.E(X)

        z = self.reparameterize(mu, log_var)
        y = self.M2(z, latent_ID_target)
        Fake = self.D(y)
        if not self.isTrain:
            return Fake

        loss = self.loss_function(Fake, img_target, mu, log_var, weight = 1)
        return loss

    def save(self, epoch_label):
        self.save_net(self.M1, 'M1', epoch_label, self.gpu_ids)
        self.save_net(self.E, 'E', epoch_label, self.gpu_ids)
        self.save_net(self.M2, 'M2', epoch_label, self.gpu_ids)
        self.save_net(self.D, 'D', epoch_label, self.gpu_ids)
