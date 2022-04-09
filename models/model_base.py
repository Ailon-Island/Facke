import os
import torch
from torch import nn
class model_base(nn.Module):
    def __init__(self):
        super(model_base, self).__init__()

    def init(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoint_dir, opt.model_name)

    def forward(self):
        pass

    def save_net(self, net, epoch_label, net_name, gpu_ids):
        save_epoch_dir = epoch_label
        save_dir = os.path.join(self.save_dir, save_epoch_dir)
        save_filename = net_name
        save_path = os.path.join(save_dir, save_filename)
        torch.save(net.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            net.cuda()

