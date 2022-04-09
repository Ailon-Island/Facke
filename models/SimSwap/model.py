from . import networks
from ..model_base import ModelBase

class GAN(ModelBase):
    def __init__(self):
        super(GAN, self).__init__()


    def init(self, opt):
        ModelBase.init()


        G = networks.Generator()
        D1 = networks.Discriminator()
        D2 = networks.Discriminator()


    def forward(self, x):
        pass