import torch
from torch import nn
from torch.nn.utils import weight_norm


class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c, h // self.bs, self.bs, w // self.bs, self.bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(n, c * (self.bs ** 2), h // self.bs, w // self.bs)
        return x


class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        n, c, h, w = x.size()
        x = x.view(n, self.bs, self.bs, c // (self.bs ** 2), h, w)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
        x = x.view(n, c // (self.bs ** 2), h * self.bs, w * self.bs)
        return x


class IDAct(nn.Module):
    def forward(self, input):
        return input


class NormConv2d(nn.Module):
    """
    Convolutional layer with l2 weight normalization and learned scaling parameters
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.beta = nn.Parameter(
            torch.zeros([1, out_channels, 1, 1], dtype=torch.float32)
        )
        self.gamma = nn.Parameter(
            torch.ones([1, out_channels, 1, 1], dtype=torch.float32)
        )
        self.conv = weight_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            name="weight",
        )

    def forward(self, x):
        # weight normalization
        # self.conv.weight = normalize(self.conv.weight., dim=[0, 2, 3])
        out = self.conv(x)
        out = self.gamma * out + self.beta
        return out


class Downsample(nn.Module):
    def __init__(self, channels, out_channels=None, conv_layer=NormConv2d):
        super().__init__()
        if out_channels == None:
            self.down = conv_layer(
                channels, channels, kernel_size=3, stride=2, padding=1
            )
        else:
            self.down = conv_layer(
                channels, out_channels, kernel_size=3, stride=2, padding=1
            )

    def forward(self, x):
        return self.down(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, subpixel=True, conv_layer=NormConv2d):
        super().__init__()
        if subpixel:
            self.up = conv_layer(in_channels, 4 * out_channels, 3, padding=1)
            self.op2 = DepthToSpace(block_size=2)
        else:
            # channels have to be bisected because of formely concatenated skips connections
            self.up = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=3, stride=2, padding=1
            )
            self.op2 = IDAct()

    def forward(self, x):
        out = self.up(x)
        out = self.op2(out)
        return out


class VUnetResnetBlock(nn.Module):
    """
    Resnet Block as utilized in the vunet publication
    """

    def __init__(
        self,
        out_channels,
        use_skip=False,
        kernel_size=3,
        activate=True,
        conv_layer=NormConv2d,
        gated=False,
        final_act=False,
        dropout_prob=0.0,
    ):
        """

        :param n_channels: The number of output filters
        :param process_skip: the factor between output and input nr of filters
        :param kernel_size:
        :param activate:
        """
        super().__init__()
        self.dout = nn.Dropout(p=dropout_prob)
        self.use_skip = use_skip
        self.gated = gated
        if self.use_skip:
            self.conv2d = conv_layer(
                in_channels=2 * out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
            self.pre = conv_layer(
                in_channels=out_channels, out_channels=out_channels, kernel_size=1,
            )
        else:
            self.conv2d = conv_layer(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )

        if self.gated:
            self.conv2d2 = conv_layer(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
            self.dout2 = nn.Dropout(p=dropout_prob)
            self.sigm = nn.Sigmoid()
        if activate:
            self.act_fn = nn.LeakyReLU() if final_act else nn.ELU()
        else:
            self.act_fn = IDAct()

    def forward(self, x, a=None):
        x_prc = x

        if self.use_skip:
            assert a is not None
            a = self.pre(a)
            x_prc = torch.cat([x_prc, a], dim=1)

        x_prc = self.act_fn(x_prc)
        x_prc = self.dout(x_prc)
        x_prc = self.conv2d(x_prc)

        if self.gated:
            x_prc = self.act_fn(x_prc)
            x_prc = self.dout(x_prc)
            x_prc = self.conv2d2(x_prc)
            a, b = torch.split(x_prc, 2, 1)
            x_prc = a * self.sigm(b)

        return x + x_prc
