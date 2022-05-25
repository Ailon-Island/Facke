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

class ILVR(nn.Module):
    def __init__(self, opt=None):
        super(ILVR, self).__init__()
        
        self.opt = opt
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        self.device = device

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



    def forward(self, img_source, img_target):
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




