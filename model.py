import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import *

class Generator(nn.Module):
    def __init__(self, in_chs=3, out_chs=3):
        super(Generator, self).__init__()

        self.enc1 = CBR2d(in_chs=in_chs, out_chs=64, batchnorm=False, relu=0.2)
        self.enc2 = CBR2d(in_chs=64, out_chs=128, relu=0.2)
        self.enc3 = CBR2d(in_chs=128, out_chs=256, relu=0.2)
        self.enc4 = CBR2d(in_chs=256, out_chs=512, relu=0.2)
        self.enc5 = CBR2d(in_chs=512, out_chs=512, relu=0.2)
        self.enc6 = CBR2d(in_chs=512, out_chs=512, relu=0.2)
        self.enc7 = CBR2d(in_chs=512, out_chs=512, relu=0.2)
        self.enc8 = CBR2d(in_chs=512, out_chs=512, relu=0.2)

        self.dec8 = CBDR2d(in_chs=512, out_chs=512)
        self.dec7 = CBDR2d(in_chs=1024, out_chs=512)
        self.dec6 = CBDR2d(in_chs=1024, out_chs=512)
        self.dec5 = CBR2d(in_chs=1024, out_chs=512, path='dec')
        self.dec4 = CBR2d(in_chs=1024, out_chs=256, path='dec')
        self.dec3 = CBR2d(in_chs=512, out_chs=128, path='dec')
        self.dec2 = CBR2d(in_chs=256, out_chs=64, path='dec')

        layers = []
        layers += [nn.ConvTranspose2d(in_channels=128, out_channels=out_chs,
                                kernel_size=4, stride=2, padding=1, bias=True)]
        layers += [nn.Sigmoid()]
        self.final = nn.Sequential(*layers)

    def forward(self, x):

        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        enc6 = self.enc6(enc5)
        enc7 = self.enc7(enc6)
        enc8 = self.enc8(enc7)

        dec8 = self.dec8(enc8)
        dec8 = torch.cat((dec8, enc7), dim=1)
        dec7 = self.dec7(dec8)
        dec7 = torch.cat((dec7, enc6), dim=1)
        dec6 = self.dec6(dec7)
        dec6 = torch.cat((dec6, enc5), dim=1)
        dec5 = self.dec5(dec6)
        dec5 = torch.cat((dec5, enc4), dim=1)
        dec4 = self.dec4(dec5)
        dec4 = torch.cat((dec4, enc3), dim=1)
        dec3 = self.dec3(dec4)
        dec3 = torch.cat((dec3, enc2), dim=1)
        dec2 = self.dec2(dec3)
        dec2 = torch.cat((dec2, enc1), dim=1)

        output = self.final(dec2)
        return output

class Discriminator(nn.Module):
    def __init__(self, in_chs=6, out_chs=1):
        super(Discriminator, self).__init__()

        self.enc1 = CBR2d(in_chs=in_chs, out_chs=64, batchnorm=False, relu=0.2)
        self.enc2 = CBR2d(in_chs=64, out_chs=128, relu=0.2)
        self.enc3 = CBR2d(in_chs=128, out_chs=256, relu=0.2)
        self.enc4 = CBR2d(in_chs=256, out_chs=512, relu=0.2)

        layers = []
        layers += [nn.Conv2d(in_channels=512, out_channels=out_chs,
                                kernel_size=4, stride=2, padding=1, bias=True)]
        layers += [nn.Sigmoid()]
        self.final = nn.Sequential(*layers)

    def forward(self, x):

        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        output = self.final(enc4)

        return output
