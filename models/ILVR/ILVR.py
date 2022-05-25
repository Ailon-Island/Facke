import torch.cuda
from torch import nn

from utils.guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from utils.resizer import Resizer
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
        self.model.to(device)
        if opt.fp16:
            self.model.convert_to_fp16()
        self.model.eval()

        assert math.log(opt.down_N, 2).is_integer()

        shape = (opt.batchSize, 3, opt.image_size, opt.image_size)
        shape_d = (opt.batchSize, 3, int(opt.image_size / opt.down_N), int(opt.image_size / opt.down_N))
        down = Resizer(shape, 1 / opt.down_N).to(next(self.model.parameters()).device)
        up = Resizer(shape_d, opt.down_N).to(next(self.model.parameters()).device)
        self.resizers = (down, up)

        schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)



    def swap(self, img_source, img_target):
        opt = self.opt
        device = self.device
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


    def forward(self, img):
        pass




