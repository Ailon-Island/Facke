import torch
from torch import nn
from torchvision import transforms

###############################################################################
# Code from
# https://github.com/neuralchen/SimSwap
###############################################################################
class InstanceNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        """
            @notice: avoid in-place ops.
            https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3
        """
        super(InstanceNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        x   = x - torch.mean(x, (2, 3), True)
        tmp = torch.mul(x, x) # or x ** 2
        tmp = torch.rsqrt(torch.mean(tmp, (2, 3), True) + self.epsilon)
        return x * tmp



class AdaIn(nn.Module):
    def __init__(self, latent_size, dim):
        super(AdaIn, self).__init__()

        self.linear = nn.Linear(latent_size, dim * 2) # one for mean and another for var


    def forward(self, x, latent_ID):
        ID_style = self.linear(latent_ID)
        ID_style = ID_style.view(-1, 2, x.size(1), 1, 1)

        x = x * (ID_style[:, 0] * 1 + 1.) + ID_style[:, 1] * 1 # why *1 ?
        return x



class IDBlock(nn.Module):
    def __init__(self, dim, latent_size, padding_mode, activation=nn.ReLU(True)):
        super(IDBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(padding=1) if padding_mode == 'reflect' else
            nn.ReplicationPad2d(padding=1) if padding_mode == 'replicate' else
            nn.ZeroPad2d(padding=1), # padding_mode == 'zero'
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3),
            InstanceNorm()
        )
        self.adain1 = AdaIn(latent_size, dim)

        self.act = activation

        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(padding=1) if padding_mode == 'reflect' else
            nn.ReplicationPad2d(padding=1) if padding_mode == 'replicate' else
            nn.ZeroPad2d(padding=1), # padding_mode == 'zero'
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3),
            InstanceNorm()
        )
        self.adain2 = AdaIn(latent_size, dim)


    def forward(self, x, latent_id):
        y = self.conv1(x)
        y = self.adain1(y, latent_id)
        y = self.act(y)
        y = self.conv2(y)
        y = self.adain2(y, latent_id)
        y = x + y
        return y



class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, latent_size, num_ID_blocks=9,
                 norm=nn.BatchNorm2d,
                 padding_mode='reflect', activation = nn.ReLU(inplace=True)):
        super(Generator, self).__init__()

        # first convolution
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(padding=3) if padding_mode == 'reflect' else
            nn.ReplicationPad2d(padding=3) if padding_mode == 'replicate' else
            nn.ZeroPad2d(padding=3), # padding_mode == 'zero'
            nn.Conv2d(in_channels, out_channels=64, kernel_size=7),
            norm(num_features=64),
            activation
        )

        # downsampling
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            norm(num_features=128),
            activation
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            norm(num_features=256),
            activation
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            norm(num_features=512),
            activation
        )

        # ID-blocks
        ID_blocks = []
        for i in range(num_ID_blocks):
            ID_blocks += [IDBlock(512, latent_size, padding_mode, activation)]
        self.ID_blocks = nn.Sequential(*ID_blocks)

        upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        # upsampling
        self.up1 = nn.Sequential(
            upsample,
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            activation
        )
        self.up2 = nn.Sequential(
            upsample,
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            activation
        )
        self.up3 = nn.Sequential(
            upsample,
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            activation
        )

        # last convolution
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(padding=3) if padding_mode == 'reflect' else
            nn.ReplicationPad2d(padding=3) if padding_mode == 'replicate' else
            nn.ZeroPad2d(padding=3),  # padding_mode == 'zero'
            nn.Conv2d(64, out_channels, kernel_size=7),
            norm(num_features=out_channels),
            nn.Tanh()
        )


    def forward(self, x, latent_id):
        x = self.conv1(x)

        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)

        for ID_block in self.ID_blocks:
            x = ID_block(x, latent_id)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)

        x = self.conv2(x)

        x = (x + 1) / 2

        return x



class Discriminator(nn.Module):
    def __init__(self, in_channels, norm=nn.BatchNorm2d, activation=nn.LeakyReLU(0.2, True), use_sigmoid=False):
        super(Discriminator, self).__init__()

        kernel_size = 4
        padding = 1

        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=kernel_size, stride=2, padding=padding),
            activation
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=kernel_size, stride=2, padding=padding),
            norm(128),
            activation
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=kernel_size, stride=2, padding=padding),
            norm(256),
            activation
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=kernel_size, stride=2, padding=padding),
            norm(512),
            activation
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=kernel_size, stride=1, padding=padding),
            norm(512),
            activation
        )

        if use_sigmoid:
            self.conv2 = nn.Sequential(
                nn.Conv2d(512, 1, kernel_size=kernel_size, stride=1, padding=padding),
                nn.Sigmoid()
            )
        else:
            self.conv2 = nn.Sequential(
                nn.Conv2d(512, 1, kernel_size=kernel_size, stride=1, padding=padding)
            )


    def forward(self, x):
        out = []

        x = self.down1(x)
        out += [x]
        x = self.down2(x)
        out += [x]
        x = self.down3(x)
        out += [x]
        x = self.down4(x)
        out += [x]

        x = self.conv1(x)
        out += [x]
        x = self.conv2(x)
        out += [x]

        return out

