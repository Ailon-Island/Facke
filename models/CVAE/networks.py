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
        y = self.conv1(x)
        y = self.adain1(y, latent_id)
        y = self.act(y)
        y = self.conv2(y)
        y = self.adain2(y, latent_id)
        y = x + y
        return y

class Encoder(nn.Module):
    def __init__(self, in_channels, latent_size = 512,
                 norm = nn.BatchNorm2d,
                 padding_mode = 'reflect', activation = nn.ReLU(inplace=True)):
        super(Encoder, self).__init__()

         # first convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=64, kernel_size=3, stride = 2, padding =1),
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

        self.encoder = nn.Sequential(self.conv1, self.down1,self.down2,self.down3)
        self.mu = nn.Linear(512, latent_size)
        self.log = nn.Linear(512, latent_size)
    def forward(self, x):
        print(x.shape)
        x = self.encoder(x)
        print(x.shape)
        mu = self.mu(x)
        log_var = self.log(x)
        return [mu, log_var]

class Decoder(nn.Module):
    def __init__(self, out_channels = 3, activation=nn.LeakyReLU(0.2, True)):
        super(Decoder,self).__init__()
        upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up1 = nn.Sequential(
            upsample,
            nn.Conv2d(in_channels= 512, out_channels = 256, kernel_size= 3, stride = 1, padding  = 1),
            nn.BatchNorm2d(256),
            activation
        )
        self.up2 = nn.Sequential(
            upsample,
            nn.Conv2d(in_channels= 256, out_channels = 128, kernel_size= 3, stride = 1, padding  = 1),
            nn.BatchNorm2d(128),
            activation
        )
        self.up3 = nn.Sequential(
            upsample,
            nn.Conv2d(in_channels= 128, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(64),
            activation
        )
        self.encode = nn.Sequential(self.up1, self.up2, self.up3)

        self.conv = nn.Sequential(
            nn.ReflectionPad2d(padding=3),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(64, out_channels, kernel_size=7),
            nn.BatchNorm2d(num_features=out_channels),
            nn.Tanh()
        )
    def forward(self, x):
        x = self.encode(x)
        return self.conv(x)

class Merge_Image(nn.Module):
    def __init__(self, in_channels, img_size=224):
        super(Merge_Image, self).__init__()
        self.img_size = img_size
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.emb = nn.Conv2d(in_channels, 1, kernel_size=1)
    def forward(self, Img_Source, Img_Target):
        print(Img_Source.shape, Img_Target.shape)

        X = self.conv(Img_Source)

        y = self.emb(Img_Target)
        print(X.shape,y.shape)
        # y = y.view(-1, self.img_size, self.img_size).unsqueeze(1)
        X = torch.cat([X, y], dim = 1)
        return X

class Merge_Latent(nn.Module): # Sample_X + Y_ID -> ADIN_LATENT -> 512 * 512 ?
    def __init__(self, in_channels, out_channels=4, latent_size=512, num_ID_blocks = 9,
            norm=nn.BatchNorm2d,
                 padding_mode='reflect', activation = nn.ReLU(inplace=True)):
        super(Merge_Latent,self).__init__()
        ID_blocks = []
        for i in range(num_ID_blocks):
            ID_blocks += [IDBlock(512,latent_size,padding_mode, activation)]
        self.ID_blocks = nn.Sequential(*ID_blocks)
        self.merge_latent = nn.Linear(in_features= in_channels, out_features= latent_size * out_channels)

    def forward(self,x,latent_id):
        for ID_block in self.ID_blocks:
            x = ID_block(x, latent_id)            
        return self.merge_latent(x)
