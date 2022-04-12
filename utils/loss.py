import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

class GANLoss(nn.Module):
    def __init__(self, gan_mode, real_label=1., fake_label=0., Tensor=torch.FloatTensor, opt=None):
        super(GANLoss, self).__init__()

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



class IDLoss(nn.Module):
    def __init__(self):
        super(IDLoss, self).__init__()

        self.sim = nn.CosineSimilarity(dim=1)

    def forward(self, x, y):
        sim = self.sim(x,y)
        loss = 1 - sim
        loss = loss.sum()
        return loss



class GPLoss(nn.Module):
    def __init__(self):
        super(GPLoss, self).__init__()

    def forward(self, D, img_real, img_fake):
        # interpolation
        batch_size = img_fake.shape[0]
        alpha = torch.rand(batch_size, 1, 1, 1).expand_as(img_fake).cuda()
        img_interp = Variable(alpha * img_real + (1-alpha) * img_fake, requires_grad=True)

        # get gradients
        pred_interp = D.forward(img_interp)
        pred_interp = pred_interp[-1]
        grad = torch.autograd.grad(outputs=pred_interp,
                                   inputs=img_interp,
                                   grad_outputs=torch.ones(pred_interp.size()).cuda(),
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        # compute loss
        grad = grad.view(grad.size(0), -1)
        grad_norm = torch.norm(grad, dim=1)
        loss = torch.mean((grad_norm - 1) ** 2)

        return loss
    
    
    
class WFMLoss(nn.Module):
    def __init__(self, num_layers, num_D):
        super(WFMLoss, self).__init__()

        self.feat_weight = 4. / (num_layers + 1)
        self.D_weight = 1. / num_D
        self.diff = nn.L1Loss()


    def forward(self, feat):
        loss = 0
        for (feat_D_real, feat_D_fake) in zip(*feat):
            for (feat_layer_real, feat_layer_fake) in zip(feat_D_real, feat_D_fake):
                loss += self.diff(feat_layer_real, feat_layer_fake)
        loss = self.feat_weight * self.D_weight * loss

        return loss


