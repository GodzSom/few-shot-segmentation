import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import numpy as np

# https://colab.research.google.com/github/vsitzmann/siren/blob/master/explore_siren.ipynb#scrollTo=RYvGrugeZbW5
class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        print(self.omega_0)
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate

#Residual down sampling block for the encoder
#Average pooling is used to perform the downsampling
class Res_down(nn.Module):
    def __init__(self, channel_in, channel_out, scale = 2, first_omega_0=30, hidden_omega_0=30.):
        super(Res_down, self).__init__()

        self.conv1 = nn.Conv2d(channel_in, channel_out//2, 3, 1, 1)
        self.BN1 = nn.BatchNorm2d(channel_out//2)
        self.Sine1 = SineLayer(channel_out//2, channel_out//2, is_first=True, omega_0=first_omega_0)

        self.conv2 = nn.Conv2d(channel_out//2, channel_out, 3, 1, 1)
        self.BN2 = nn.BatchNorm2d(channel_out)
        self.Sine2 = SineLayer(channel_out, channel_out, is_first=False, omega_0=hidden_omega_0)

        self.conv3 = nn.Conv2d(channel_in, channel_out, 3, 1, 1)

        self.AvePool = nn.AvgPool2d(scale,scale)

    def forward(self, x):
        skip = self.conv3(self.AvePool(x))

        x = self.BN1(self.conv1(x))
        x = self.Sine1(x)
        x = self.AvePool(x)
        x = self.BN2(self.conv2(x))

        x = self.Sine2(x + skip)
        return x


#Residual up sampling block for the decoder
#Nearest neighbour is used to perform the upsampling
class Res_up(nn.Module):
    def __init__(self, channel_in, channel_out, scale = 2, first_omega_0=30, hidden_omega_0=30.):
        super(Res_up, self).__init__()

        self.conv1 = nn.Conv2d(channel_in, channel_out//2, 3, 1, 1)
        self.BN1 = nn.BatchNorm2d(channel_out//2)
        self.Sine1 = SineLayer(channel_out//2, channel_out//2, is_first=True, omega_0=first_omega_0)
        self.conv2 = nn.Conv2d(channel_out//2, channel_out, 3, 1, 1)
        self.BN2 = nn.BatchNorm2d(channel_out)
        self.Sine2 = SineLayer(channel_out, channel_out, is_first=False, omega_0=hidden_omega_0)

        self.conv3 = nn.Conv2d(channel_in, channel_out, 3, 1, 1)

        self.UpNN = nn.Upsample(scale_factor = scale,mode = "nearest")

    def forward(self, x):
        skip = self.conv3(self.UpNN(x))

        x = self.Sine1(self.BN1(self.conv1(x)))
        x = self.UpNN(x)
        x = self.BN2(self.conv2(x))

        x = self.Sine2(x + skip)
        return x

#Encoder block
#Built for a 64x64x3 image and will result in a latent vector of size Z x 1 x 1
#As the network is fully convolutional it will work for other larger images sized 2^n the latent
#feature map size will just no longer be 1 - aka Z x H x W
class Encoder(nn.Module):
    def __init__(self, channels, ch = 64, z = 512):
        super(Encoder, self).__init__()
        self.conv1 = Res_down(channels, ch)
        self.conv2 = Res_down(ch, 2*ch)
        self.conv3 = Res_down(2*ch, 4*ch)
        self.conv4 = Res_down(4*ch, 8*ch)
        self.conv_mu = nn.Conv2d(8*ch, z, 4, 1)
        self.conv_logvar = nn.Conv2d(8*ch, z, 4, 1)

    def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x, Train = True):
        # print(x.shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        if Train:
            mu = self.conv_mu(x)
            logvar = self.conv_logvar(x)
            x = self.sample(mu, logvar)
        else:
            x = self.conv_mu(x)
            mu = None
            logvar = None
        # print(x.shape)
        return x, mu, logvar

#Decoder block
#Built to be a mirror of the encoder block
class Decoder(nn.Module):
    def __init__(self, channels, ch = 64, z = 512):
        super(Decoder, self).__init__()
        self.conv1 = Res_up(z, ch*8, scale = 4)#4
        self.conv2 = Res_up(ch*8, ch*4)
        self.conv3 = Res_up(ch*4, ch*2)
        self.conv4 = Res_up(ch*2, ch)
        self.conv5 = Res_up(ch, ch//2)
        self.conv6 = nn.Conv2d(ch//2, channels, 3, 1, 1)
        # self.conv6 = nn.Conv2d(ch, channels, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        # assert False

        return x

#VAE network, uses the above encoder and decoder blocks
class VAE(nn.Module):
    def __init__(self, channel_in, z = 512):
        super(VAE, self).__init__()
        """Res VAE Network
        channel_in  = number of channels of the image
        z = the number of channels of the latent representation (for a 64x64 image this is the size of the latent vector)"""

        self.encoder = Encoder(channel_in, z = z)
        self.decoder = Decoder(channel_in, z = z)

    def forward(self, x, Train = True):
        encoding, mu, logvar = self.encoder(x, Train)
        recon = self.decoder(encoding)
        return recon, mu, logvar
