###############################################################################
# Code adapted and modified from
# https://github.com/jychoi118/ilvr_adm
###############################################################################
import copy
import functools
import os

import torch.cuda
from torch import nn
from torch.optim import AdamW

from utils.guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from utils.guided_diffusion.resample import create_named_schedule_sampler
from utils.guided_diffusion.fp16_util import MixedPrecisionTrainer
from utils.resizer import Resizer
from utils.guided_diffusion.resample import LossAwareSampler, UniformSampler
from utils.guided_diffusion.nn import update_ema

import math

from ..model_base import ModelBase

class ILVR(ModelBase):
    def __init__(self):
        super(ILVR, self).__init__()


    def init(self, opt):
        if opt.verbose:
            print('Initializing DDPM for ILVR...')
        
        self.opt = opt

        self.isTrain = opt.isTrain
        self.gpu_ids = opt.gpu_ids

        self.iter = 0

        self.device = torch.device(self.gpu_ids[0])

        model, diffusion = create_model_and_diffusion(
            **args_to_dict(opt, model_and_diffusion_defaults().keys())
        )
        self.model = model
        self.diffusion = diffusion
        self.model.load_state_dict(torch.load(opt.DDPM_pth))
        self.model.to(self.device)
        if opt.fp16:
            self.model.convert_to_fp16()
        self.model.eval()

        assert math.log(opt.down_N, 2).is_integer()

        shape = (opt.batchSize, 3, opt.image_size, opt.image_size)
        shape_d = (opt.batchSize, 3, int(opt.image_size / opt.down_N), int(opt.image_size / opt.down_N))
        down = Resizer(shape, 1 / opt.down_N).to(next(self.model.parameters()).device)
        up = Resizer(shape_d, opt.down_N).to(next(self.model.parameters()).device)
        self.resizers = (down, up)

        self.schedule_sampler = create_named_schedule_sampler(opt.schedule_sampler, diffusion)

        params = list(self.model.parameters())

        self.lr = opt.lr
        self.weight_decay = opt.weight_decay
        self.optim = AdamW(params, lr=self.lr, weight_decay=self.weight_decay)

        self.ema_rate = (
            [opt.ema_rate]
            if isinstance(opt.ema_rate, float)
            else [float(x) for x in opt.ema_rate.split(",")]
        )
        self.ema_params = [
            copy.deepcopy(params)
            for _ in range(len(self.ema_rate))
        ]


    def swap(self, img_source, img_target):
        opt = self.opt
        device = img_source.device
        diffusion = self.diffusion
        model = self.model
        resizers = self.resizers

        shape = (opt.batchSize, 3, opt.image_size, opt.image_size)

        noised_source = diffusion.q_sample(img_source,
                                           t=torch.tensor([int(opt.timestep_respacing) - 1] * shape[0], device=device),
                                           noise=torch.randn_like(img_source, device=device))

        model_kwargs = {}
        model_kwargs['ref_img'] = img_target
        sample = diffusion.p_sample_loop(
            model,
            shape,
            noise=noised_source,
            clip_denoised=opt.clip_denoised,
            model_kwargs=model_kwargs,
            resizers=resizers,
            range_t=opt.range_t,
        )

        return sample


    def forward(self, img, cond = None):
        #self.optim.zero_grad()
        # for i in range(0, img.shape[0], self.microimg):
        device = img.device

        if cond is None:
            cond = {}
        else:
            cond = {
                k: v.to(device)
                for k, v in cond.items()
            }

        t, weights = self.schedule_sampler.sample(img.shape[0], device)

        # compute_losses = functools.partial(
        #     self.diffusion.training_losses,
        #     self.ddp_model,
        #     micro,
        #     t,
        #     model_kwargs=micro_cond,
        # )

        losses = self.diffusion.training_losses(
            model   =   self.model,
            x_start =   img,
            t = t,
            model_kwargs = cond,
        )

        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, losses["loss"].detach()
            )

        losses = {k:(v * weights).mean() for k, v in losses.items()}
        #loss.backward()

        return losses


    def save(self, epoch_label):
        self.save_net(self.model,  'DDPM', epoch_label, self.gpu_ids)


    def load(self, epoch_label):
        self.load_net(self.model, 'DDPM', epoch_label, self.gpu_ids)


    def update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, list(self.model.parameters()), rate=rate)

