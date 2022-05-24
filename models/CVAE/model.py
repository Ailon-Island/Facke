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

        if torch.cuda.is_available():
            device = torch.device(self.gpu_ids[0])
        else:
            device = torch.device("cpu")

        #Merge 1

        self.M1 = networks.Merge_Image(in_channels = 3)
        self.M1 = self.M1.to(device)
        #Encoder
        self.E = networks.Encoder(in_channels= 4)
        self.E = self.E.to(device)

        #Encoder -> MERGE -> DECODER
        self.M2 = networks.Merge_Distribution(in_channels=512)
        self.M2 = self.M2.to(device)
    
        #Decoder
        self.D = networks.Decoder()
        self.D = self.D.to(device)

        # loss functions
        self.loss_names = ['Rec', 'KL']
        self.Recloss = nn.MSELoss()
        self.KLloss = loss.KLLoss(Weight= 0.00025)

        # optimizers
        params = list(self.M1.parameters()) + list(self.E.parameters()) + list(self.M2.parameters()) + list(self.D.parameters())
        self.optim = torch.optim.Adam(params, lr = opt.lr, betas=(opt.beta1, 0.999))

    # def loss_function(self, recons, input, mu, log_var, weight):
    #     recons_loss = F.mse_loss(recons, input)
    #     kld_loss = torch.mean(-0.5 * torch.sum(1+ log_var - mu**2 - log_var.exp(),dim=1),dim = 0)
    #     loss = recons_loss + weight * kld_loss
    #     return loss

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.rand_like(std)
        return eps* std + mu

    def forward(self, img_source, img_target, latent_ID, latent_ID_target):

        X = self.M1(img_source, img_target)

        mu, log_var = self.E(X)
        # print("=========In CVAE.forward=======")
        # print("MU",mu.shape)

        mu, log_var = self.M2(mu,log_var, latent_ID_target)


        z = self.reparameterize(mu, log_var)
        # print("z", z.shape)
        # print("LAtent_ID_target", latent_ID_target.shape)
        # y = self.M2(z, latent_ID_target)
        Fake = self.D(z)
        if not self.isTrain:
            return Fake

        loss_Rec = self.Recloss(Fake, img_source)
        loss_KL = self.KLloss(mu, log_var)

        return [[loss_Rec, loss_KL], Fake]

    def save(self, epoch_label):
        self.save_net(self.M1, 'M1', epoch_label, self.gpu_ids)
        self.save_net(self.E, 'E', epoch_label, self.gpu_ids)
        self.save_net(self.M2, 'M2', epoch_label, self.gpu_ids)
        self.save_net(self.D, 'D', epoch_label, self.gpu_ids)

    def update_lr(self):
        lr_decay = self.opt.lr / (self.opt.niter_decay + 1)
        lr = self.old_lr - lr_decay

        for param_group in self.optim.param_groups:
            param_group['lr'] = lr

        self.old_lr = lr