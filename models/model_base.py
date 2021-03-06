###############################################################################
# Code from
# https://github.com/neuralchen/SimSwap
###############################################################################
import os
import sys
import torch
from torch import nn
from utils import utils

class ModelBase(nn.Module):
    def __init__(self):
        super(ModelBase, self).__init__()


    def init(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)


    def forward(self):
        pass


    def save_net(self, net, net_name, epoch_label, gpu_ids):
        save_dir = os.path.join(self.save_dir, net_name)
        utils.mkdirs(save_dir)
        save_path = os.path.join(save_dir, epoch_label+".pth")

        torch.save(net.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            net.cuda()


    def load_net(self, net, net_name, epoch_label, gpu_ids, save_dir=''):
        if not save_dir:
            save_dir = self.save_dir
        save_path = os.path.join(save_dir, net_name, epoch_label+'.pth')

        if not os.path.isfile(save_path):
            print('{} does not exist!'.format(save_path))
        else:
            net.load_state_dict(torch.load(save_path))



