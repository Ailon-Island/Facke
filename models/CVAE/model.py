import  os
import sys
import time



root_path = os.path.join("..", "..")
sys.path.append(root_path)
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
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
        self.img_size = opt.image_size
        self.iter = 0

        if torch.cuda.is_available():
            device = torch.device(self.gpu_ids[0])
        else:
            device = torch.device("cpu")


        self.INnorm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # normalization of ImageNet
        
        #Merge 1

        self.M1 = networks.Merge_Image(in_channels=512, img_size = self.img_size)
        self.M1 = self.M1.to(device)
        #Encoder
        self.E = networks.Encoder(in_channels= 4, out_channels = 512)
        self.E = self.E.to(device)

        #Encoder -> MERGE -> DECODER
        self.M2 = networks.Merge_Distribution(latent_size = 512, num_ID_blocks = 9)
        self.M2 = self.M2.to(device)
    
        #Decoder
        self.D = networks.Decoder(in_channels = 512, out_channels = 3)
        self.D = self.D.to(device)
        
        # self.G = networks.Generator(in_channels=3,out_channels=3,latent_size=512,num_ID_blocks = 0)
        # self.G = self.G.to(device)

        # ID network
        self.ID_extract = IDExtractor(self.opt)
        self.ID_extract.eval()

        # loss functions
        self.loss_names = ['Rec', 'KL', 'ID']
        self.Recloss = nn.L1Loss()
        self.KLloss = loss.KLLoss(Weight=opt.lambda_KL)
        self.IDloss = loss.IDLoss()
        
        # optimizers
        params = list(self.M1.parameters()) + list(self.E.parameters()) + list(self.M2.parameters()) + list(self.D.parameters())
        self.optim = torch.optim.Adam(params, lr = opt.lr, betas=(opt.beta1, 0.999))
        
        # params = list(self.G.parameters())
        # self.optim = torch.optim.Adam(params, lr = opt.lr, betas=(opt.beta1, 0.999))

        self.old_lr = opt.lr

        if opt.verbose:
            print("CVAE model initiated.")
    # def loss_function(self, recons, input, mu, log_var, weight):
    #     recons_loss = F.mse_loss(recons, input)
    #     kld_loss = torch.mean(-0.5 * torch.sum(1+ log_var - mu**2 - log_var.exp(),dim=1),dim = 0)
    #     loss = recons_loss + weight * kld_loss
    #     return loss

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.rand_like(std)
        return eps* std + mu

    def forward(self, Img, latent_ID, is_same_ID=False):

        loss_Rec, loss_KL, loss_ID = torch.tensor(0., device=Img.device), torch.tensor(0., device=Img.device), torch.tensor(0., device=Img.device)

        X = self.M1(Img, latent_ID)

        # mu, log_var, X_ID = self.E(X)
        mu, log_var = self.E(X)
        z = self.reparameterize(mu, log_var)
        Inject_z = self.M2(z, latent_ID)
        # print("=========In CVAE.forward=======")
        # print("MU",mu.shape)
        # print("X_ID",X_ID.shape)
        # Inject_mu, Inject_log_var = self.M2(mu,log_var, latent_ID_target)


        # z = self.reparameterize(Inject_mu, Inject_log_var)
        # print("z", z.shape)
        # print("LAtent_ID_target", latent_ID_target.shape)
        # y = self.M2(z, latent_ID_target)
        Fake = self.D(Inject_z)
        # Fake = self.D(X_ID)
        if not self.isTrain:
            return Fake

        Fake = self.INnorm(Fake)

        latent_ID_fake = self.ID_extract(Fake)
        loss_ID = self.IDloss(latent_ID_fake, latent_ID)
        loss_ID *= self.opt.lambda_id
        loss_Rec = self.Recloss(Fake, Img)
        if not is_same_ID:
            loss_Rec *= self.opt.lambda_rec_swap
        loss_KL = self.KLloss(mu, log_var)

        return [[loss_Rec, loss_KL, loss_ID], Fake]
        
        # img_fake = self.G(img_target,latent_ID)
        # if not self.isTrain:
        #     return img_fake
        # img_fake = self.INnorm(img_fake)
        # loss_Rec = self.Recloss(img_fake, img_target)
        # loss_KL = loss_Rec * 0
        # return [[loss_Rec, loss_KL], img_fake]



    def save(self, epoch_label):
        
        self.save_net(self.M1, 'M1', epoch_label, self.gpu_ids)
        self.save_net(self.E, 'E', epoch_label, self.gpu_ids)
        self.save_net(self.M2, 'M2', epoch_label, self.gpu_ids)
        self.save_net(self.D, 'D', epoch_label, self.gpu_ids)

        # self.save_net(self.G, 'G', epoch_label, self.gpu_ids)
    def update_lr(self):
        lr_decay = self.opt.lr / (self.opt.niter_decay + 1)
        lr = self.old_lr - lr_decay

        for param_group in self.optim.param_groups:
            param_group['lr'] = lr

        self.old_lr = lr