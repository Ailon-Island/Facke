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


class SimSwapGAN(ModelBase):
    def __init__(self):
        super(SimSwapGAN, self).__init__()


    def init(self, opt):
        if opt.verbose:
            print("Initializing SimSwap model...")
        ModelBase.init(self, opt)

        self.isTrain = opt.isTrain
        self.gpu_ids = opt.gpu_ids

        self.iter = 0

        device = torch.device(self.gpu_ids[0])

        # Generator
        self.G = networks.Generator(in_channels=3, out_channels=3, latent_size=512, num_ID_blocks=9)
        self.G = self.G.to(device)

        # ID network
        self.ID_extract = IDExtractor(self.opt)
        self.ID_extract.eval()

        ############### if not training, only Generator needed ###############
        if not self.isTrain:
            # load G only
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_net(self.G, 'G', opt.epoch_label, pretrained_path)
            return
        ######################################################################

        ### training ###
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

            self.load_net(self.G, 'G', opt.epoch_label, pretrained_path)
            self.load_net(self.D1, 'D1', opt.epoch_label, pretrained_path)
            self.load_net(self.D2, 'D2', opt.epoch_label, pretrained_path)

        # loss functions
        self.loss_names = ['D_real', 'D_fake', 'D_GP', 'G_GAN', 'G_FM', 'G_ID', 'G_rec']
        self.IDloss = loss.IDLoss()
        self.Recloss = nn.L1Loss()
        self.GANloss = loss.GANLoss(opt.gan_mode, Tensor=self.Tensor, opt=opt)
        self.GPloss = loss.GPLoss()
        if not opt.no_ganFeat_loss:
            self.FMloss = loss.FMLoss( opt)

        # optimizers
        params = list(self.G.parameters())
        self.optim_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

        params = list(self.D1.parameters()) + list(self.D2.parameters())
        self.optim_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

        self.old_lr = opt.lr

        if opt.verbose:
            print("SimSwap model initiated.")



    def forward(self, img_source, img_target, latent_ID, latent_ID_target):
        # loss initialization
        loss_D_real, loss_D_fake, loss_D_GP = 0, 0, 0
        loss_G_GAN, loss_G_FM, loss_G_ID, loss_G_rec =  0, 0, 0, 0

        # generate fake image
        img_fake = self.G.forward(img_target, latent_ID)
        if not self.isTrain:
            return img_fake

        img_fake = self.ID_extract.INnorm(img_fake)

        img_fake_down = self.downsample(img_fake)
        img_target_down = self.downsample(img_target)

        # D fake
        #t1 = time.time()
        feat_D1_fake = self.D1.forward(img_fake.detach())
        feat_D2_fake = self.D2.forward(img_fake_down.detach())
        pred_D_fake = [feat_D1_fake, feat_D2_fake]
        #print(time.time() - t1)

        loss_D_fake = self.GANloss(pred_D_fake, is_real=False, forD=True)

        # D real (target)
        feat_D1_real = self.D1.forward(img_target)
        feat_D2_real = self.D2.forward(img_target_down)
        pred_D_real = [feat_D1_real, feat_D2_real]
        feat_D_real = [feat_D1_real, feat_D2_real]

        loss_D_real = self.GANloss(pred_D_real, is_real=True, forD=True)

        # D GP
        loss_D_GP = self.GPloss(self.D1, img_target, img_fake.detach())
        loss_D_GP += self.GPloss(self.D2, img_target_down, img_fake_down.detach())
        loss_D_GP *= self.opt.lambda_GP

        # G GAN
        feat_D1_fake = self.D1.forward(img_fake)
        feat_D2_fake = self.D2.forward(img_fake_down)
        pred_D_fake = [feat_D1_fake, feat_D2_fake]
        feat_D_fake = [feat_D1_fake, feat_D2_fake]

        loss_G_GAN = self.GANloss(pred_D_fake, is_real=True, forD=False)

        # G GAN weak feat match
        if not self.opt.no_ganFeat_loss:
            loss_G_FM = self.FMloss([feat_D_real, feat_D_fake])
            loss_G_FM *= self.opt.lambda_FM
        else:
            loss_G_FM = loss_G_GAN * 0 # whatever, get a zero tensor

        # G ID
        latent_ID_fake = self.ID_extract(img_fake)

        loss_G_ID = self.IDloss(latent_ID_fake, latent_ID)
        loss_G_ID *= self.opt.lambda_id

        # G reconstruction
        loss_G_rec = self.Recloss(img_fake, img_target)
        loss_G_rec *= self.opt.lambda_rec

        if self.training:
            return [[loss_D_real, loss_D_fake, loss_D_GP, loss_G_GAN, loss_G_FM, loss_G_ID, loss_G_rec], img_fake]
        else:
            return [[loss_D_real.detach(), loss_D_fake.detach(), loss_D_GP.detach(), loss_G_GAN.detach(), loss_G_FM.detach(), loss_G_ID.detach(), loss_G_rec.detach()], img_fake.detach()]
        # self.loss_names = ['D_real', 'D_fake', 'D_GP', 'G_GAN', 'G_FM', 'G_ID', 'G_rec']


    def save(self, epoch_label):
        self.save_net(self.G,  'G', epoch_label,self.gpu_ids)
        self.save_net(self.D1, 'D1', epoch_label, self.gpu_ids)
        self.save_net(self.D2, 'D2', epoch_label, self.gpu_ids)


    def load(self, epoch_label):
        self.save_net(self.G, 'G', epoch_label, self.gpu_ids)
        self.save_net(self.D1, 'D1', epoch_label, self.gpu_ids)
        self.save_net(self.D2, 'D2', epoch_label, self.gpu_ids)


    def unlock_G(self):
        params = list(self.G.parameters())
        self.optim_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))


    def update_lr(self):
        lr_decay = self.opt.lr / (self.opt.niter_decay + 1)
        lr = self.old_lr - lr_decay

        for param_group in self.optim_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optim_G.param_groups:
            param_group['lr'] = lr

        self.old_lr = lr