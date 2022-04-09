import torch
from torch import nn
import torch.nn.functional as F

class LossBase(nn.Module):
    def __init__(self):
        super(LossBase, self).__init__()


    def loss(self, input):
        pass


    def __call__(self, input):
        return self.loss(input)



class GANloss(LossBase):
    def __init__(self, gan_mode, real_label=1., fake_label=0., Tensor=torch.FloatTensor, opt=None):
        super(GANloss, self).__init__()

        self.gan_mode = gan_mode
        self.real_label = real_label
        self.fake_label = fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = Tensor
        self.opt = opt


    def get_target_tensor(self, input, is_real):
        if is_real:
            if self.real_label_tensor == None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else: # fake
            if self.fake_label_tensor == None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)


    def get_zero_tensor(self, input):
        if self.zero_tensor == None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)


    def loss(self, input, is_real, forD=True):
        if self.gan_mode == 'original': # cross entropy
            target_tensor = self.get_target_tensor(input, is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, is_real) # MSE loss
            loss = F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if forD:
                if is_real:
                    min = torch.min(input - 1, self.get_zero_tensor(input))
                else:
                    min = torch.min(- input - 1, self.get_zero_tensor(input))
                loss = -torch.mean(min)
            else: # wgan
                if is_real:
                    loss = -input.mean()
                else:
                    loss = input.mean()

        return loss


    def __call__(self, input, is_real, forD=True):
        if isinstance(input, list):
            loss = 0
            for pred in input:
                if isinstance(pred, list):
                    pred = pred[-1]

                loss_tensor = self.loss(pred, is_real, forD)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0) # what is this bs?
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss

            return loss / len(input)
        else:
            return self.loss(input, is_real, forD)
