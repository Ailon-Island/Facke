from tracemalloc import start
from turtle import forward
import torch
from torch import nn
from torchvision import transforms

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
        # print("====IN IDBLOCK =====")
        # print("ORIGIN X", x.shape)
        y = self.conv1(x)
        # print("AFTER conv1", y.shape)
        y = self.adain1(y, latent_id)
        # print("AFTER ADIN1", y.shape)
        y = self.act(y)
        y = self.conv2(y)
        # print("AFTER conv2",y.shape)
        y = self.adain2(y, latent_id)
        y = x + y
        return y

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels,
                 norm=nn.BatchNorm2d,
                 padding_mode='reflect', activation = nn.ReLU(inplace=True)):
        super(Encoder, self).__init__()

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

        self.mu = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=512, kernel_size = 3, stride = 1, padding =1),
            norm(512),
            activation
        )
        self.log_var = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=512, kernel_size = 3, stride = 1, padding =1),
            norm(512),
            activation
        )
    def forward(self, x):
        # print("=========ENCODER FORWARD=========")
        # print("BEFORE DOWNSAMPLE", x.shape)
        x = self.conv1(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        # x = torch.flatten(x,start_dim = 1)
        # print("AFTER DOWNSAMPLE", x.shape)
        mu = self.mu(x)
        log_var = self.log_var(x)
        # print("MU:", mu.shape)
        # return [mu, log_var, x]
        return [mu, log_var]

class Decoder(nn.Module):
    def __init__(self, in_channels=512, out_channels = 3, padding_mode='reflect', activation = nn.ReLU(inplace=True)):
        super(Decoder,self).__init__()
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
            nn.BatchNorm2d(num_features=out_channels),
            nn.Tanh()
        )
    def forward(self, x):
        # print("=========IN Decoder========")
        # print("ORIGIN ",x.shape)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.conv2(x)
        return (x+1)/2
        # print("AFTER inputLayer", x.shape)
        # x = x.view(-1,512,(self.img_size//32),(self.img_size//32))
        # x = self.up1(x)
        # print("AFTER up1", x.shape)
        # x = self.up2(x)
        # print("AFTER up2",x.shape)
        # x = self.up3(x)
        # print("AFTER up3",x.shape)
        # x = self.up4(x)
        # x = self.decode(x)
        # print("AFTER up4", x.shape)
        # x = self.conv2(x)
        # print("AFTER outputLayer", x.shape)
        # print("===== FINISH Decode======")
        # return (x+1)/2

class Merge_Image(nn.Module):
    def __init__(self, in_channels, activation = nn.ReLU(inplace=True)):
        super(Merge_Image, self).__init__()
        self.emb = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=3, stride =1 ,padding =1),
            nn.BatchNorm2d(1),
            activation
        )


    def forward(self, Img_Source, Img_Target):
        # print(Img_Source.shape, Img_Target.shape)

        y = self.emb(Img_Source)
        # print(X.shape,y.shape)
        # y = y.view(-1, self.img_size, self.img_size).unsqueeze(1)
        X = torch.cat([Img_Target, y], dim = 1)
        return X

class Merge_Distribution(nn.Module): # Sample_X + Y_ID -> ADIN_LATENT -> 512 ?
    def __init__(self, latent_size, num_ID_blocks = 9, norm=nn.BatchNorm2d,
                 padding_mode='reflect', activation = nn.ReLU(inplace=True)):
        super(Merge_Distribution, self).__init__()
        ID_blocks = []
        for i in range(num_ID_blocks):
            ID_blocks += [IDBlock(512, latent_size, padding_mode, activation)]
        self.ID_blocks = nn.Sequential(*ID_blocks)

    def forward(self, x, latent_id):
        for ID_block in self.ID_blocks:
            x = ID_block(x, latent_id)
        return x


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, latent_size, num_ID_blocks=1,
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

        self.mu = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=512, kernel_size = 3, stride = 1, padding =1),
            norm(512),
            activation
        )
        self.log_var = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=512, kernel_size = 3, stride = 1, padding =1),
            norm(512),
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

