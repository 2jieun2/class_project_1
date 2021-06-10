import torch
from torch import nn


def match_size(x, size):
    _, _, h1, w1, d1 = x.shape
    h2, w2, d2 = size

    while d1 != d2:
        if d1 < d2:
            x = nn.functional.pad(x, (0, 1), mode='constant', value=0)
            d1 += 1
        else:
            x = x[:, :, :, :, :d2]
            break
    while w1 != w2:
        if w1 < w2:
            x = nn.functional.pad(x, (0, 0, 0, 1), mode='constant', value=0)
            w1 += 1
        else:
            x = x[:, :, :, :w2, :]
            break
    while h1 != h2:
        if h1 < h2:
            x = nn.functional.pad(x, (0, 0, 0, 0, 0, 1), mode='constant', value=0)
            h1 += 1
        else:
            x = x[:, :, :h2, :, :]
            break
    return x


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout=False):
        super().__init__()
        layers = [
            nn.Conv3d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.Conv3d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(out_c),
            nn.LeakyReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        # layers.extend([
        #     nn.Conv3d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.InstanceNorm3d(out_c),
        #     nn.LeakyReLU(inplace=True)
        # ])
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out


class UpMerge(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose3d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(out_c),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x, skip_x):
        x = self.model(x)
        if x.size() != skip_x.size():
            x = match_size(x, skip_x.shape[2:])
        x = torch.cat((x, skip_x), 1)
        return x


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        nf = 16

        self.pooling = nn.MaxPool3d(kernel_size=2)

        self.conv1 = ConvBlock(in_channels, nf)
        self.conv2 = ConvBlock(nf, nf*2)
        self.conv3 = ConvBlock(nf*2, nf*4)
        self.conv4 = ConvBlock(nf*4, nf*8)
        # self.conv5 = ConvBlock(nf*8, nf*16)
        self.conv5 = ConvBlock(nf*8, nf*8)

        self.up1 = UpMerge(nf*8, nf*8)
        self.up2 = UpMerge(nf*8*2, nf*4)
        self.up3 = UpMerge(nf*4*2, nf*2)
        self.up4 = UpMerge(nf*2*2, nf)

        # self.up6 = UpMerge(nf*16, nf*8)
        # self.conv6 = ConvBlock(nf*8*2, nf*8, 0.2)

        # self.up7 = UpMerge(nf*8, nf*4)
        # self.conv7 = ConvBlock(nf*4*2, nf*4, 0.2)

        # self.up8 = UpMerge(nf*4, nf*2)
        # self.conv8 = ConvBlock(nf*2*2, nf*2, 0.1)

        # self.up9 = UpMerge(nf*2, nf)
        # self.conv9 = ConvBlock(nf*2, nf, 0.1)

        self.out = nn.Sequential(
            # nn.Conv3d(nf, out_channels, kernel_size=1, stride=1, bias=False),
            nn.Conv3d(nf*2, out_channels, kernel_size=1, stride=1, bias=False),
            # nn.Tanh()
            # nn.ReLU()
            nn.LeakyReLU(inplace=True)
        )


    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pooling(c1)

        c2 = self.conv2(p1)
        p2 = self.pooling(c2)

        c3 = self.conv3(p2)
        p3 = self.pooling(c3)

        c4 = self.conv4(p3)
        p4 = self.pooling(c4)

        c5 = self.conv5(p4)
        #
        # u6 = self.up6(c5, c4)
        # c6 = self.conv6(u6)
        #
        # u7 = self.up7(c6, c3)
        # c7 = self.conv7(u7)
        #
        # u8 = self.up8(c7, c2)
        # c8 = self.conv8(u8)
        #
        # u9 = self.up9(c8, c1)
        # c9 = self.conv9(u9)
        #
        # out = self.out(c9)

        # c1 = self.conv1(x)
        # c2 = self.conv2(c1)
        # c3 = self.conv3(c2)
        # c4 = self.conv4(c3)
        # c5 = self.conv5(c4)
        u1 = self.up1(c5, c4)
        u2 = self.up2(u1, c3)
        u3 = self.up3(u2, c2)
        u4 = self.up4(u3, c1)
        out = self.out(u4)

        if x.size() != out.size():
            out = match_size(out, x.shape[2:])

        return u1, u2, u3, u4, out
        # return out


class TeacherUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        nf = 16

        self.pooling = nn.MaxPool3d(kernel_size=2)

        self.conv1 = ConvBlock(in_channels, nf)
        self.conv2 = ConvBlock(nf, nf*2)
        self.conv3 = ConvBlock(nf*2, nf*4)
        self.conv4 = ConvBlock(nf*4, nf*8)
        # self.conv5 = ConvBlock(nf*8, nf*16)
        self.conv5 = ConvBlock(nf*8, nf*8)

        self.up1 = UpMerge(nf*8, nf*8)
        self.up2 = UpMerge(nf*8*2, nf*4)
        self.up3 = UpMerge(nf*4*2, nf*2)
        self.up4 = UpMerge(nf*2*2, nf)

        # self.up6 = UpMerge(nf*16, nf*8)
        # self.conv6 = ConvBlock(nf*8*2, nf*8, 0.2)

        # self.up7 = UpMerge(nf*8, nf*4)
        # self.conv7 = ConvBlock(nf*4*2, nf*4, 0.2)

        # self.up8 = UpMerge(nf*4, nf*2)
        # self.conv8 = ConvBlock(nf*2*2, nf*2, 0.1)

        # self.up9 = UpMerge(nf*2, nf)
        # self.conv9 = ConvBlock(nf*2, nf, 0.1)

        self.out = nn.Sequential(
            # nn.Conv3d(nf, out_channels, kernel_size=1, stride=1, bias=False),
            nn.Conv3d(nf*2, out_channels, kernel_size=1, stride=1, bias=False),
            # nn.Tanh()
            # nn.ReLU()
            nn.LeakyReLU(inplace=True)
        )


    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pooling(c1)

        c2 = self.conv2(p1)
        p2 = self.pooling(c2)

        c3 = self.conv3(p2)
        p3 = self.pooling(c3)

        c4 = self.conv4(p3)
        p4 = self.pooling(c4)

        c5 = self.conv5(p4)
        #
        # u6 = self.up6(c5, c4)
        # c6 = self.conv6(u6)
        #
        # u7 = self.up7(c6, c3)
        # c7 = self.conv7(u7)
        #
        # u8 = self.up8(c7, c2)
        # c8 = self.conv8(u8)
        #
        # u9 = self.up9(c8, c1)
        # c9 = self.conv9(u9)
        #
        # out = self.out(c9)

        # c1 = self.conv1(x)
        # c2 = self.conv2(c1)
        # c3 = self.conv3(c2)
        # c4 = self.conv4(c3)
        # c5 = self.conv5(c4)
        u1 = self.up1(c5, c4)
        u2 = self.up2(u1, c3)
        u3 = self.up3(u2, c2)
        u4 = self.up4(u3, c1)
        out = self.out(u4)

        if x.size() != out.size():
            out = match_size(out, x.shape[2:])

        return u1, u2, u3, u4, out
        # return out


class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        nf = 16

        self.pooling = nn.MaxPool3d(kernel_size=2)

        self.conv1 = ConvBlock(in_channels, nf)
        self.conv2 = ConvBlock(nf, nf*2)
        self.conv3 = ConvBlock(nf*2, nf*4)
        self.conv4 = ConvBlock(nf*4, nf*8)
        self.conv5 = ConvBlock(nf*8, nf*16)

        # self.conv1 = ConvBlock(in_channels, nf, 0.1)
        # self.conv2 = ConvBlock(nf, nf*2, 0.1)
        # self.conv3 = ConvBlock(nf*2, nf*4, 0.2)
        # self.conv4 = ConvBlock(nf*4, nf*8, 0.2)
        # self.conv5 = ConvBlock(nf*8, nf*16, 0.3)


        self.out = nn.Sequential(
            # nn.Conv3d(nf*8, 1, kernel_size=4, padding=1, bias=False)
            nn.AdaptiveAvgPool3d(output_size=1),
            # nn.Sigmoid()
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pooling(c1)

        c2 = self.conv2(p1)
        p2 = self.pooling(c2)

        c3 = self.conv3(p2)
        p3 = self.pooling(c3)

        c4 = self.conv4(p3)
        p4 = self.pooling(c4)

        c5 = self.conv5(p4)

        out = self.out(c5)

        # c1 = self.conv1(x)
        # c2 = self.conv2(c1)
        # c3 = self.conv3(c2)
        # c4 = self.conv4(c3)
        # c5 = self.conv5(c4)
        # out = self.out(c5)

        return out