import src.network as network
import torch.nn as nn


#Model architecture inspired by STARGan implementation https://github.com/yunjey/StarGAN

class ImageNetDiscriminator(nn.Module):

    def __init__(self, dimensions, image_size=128):
        super(ImageNetDiscriminator, self).__init__()

        self.dimensions = dimensions

        self.cnv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False)

        self.cnv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)

        self.cnv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)

        self.cnv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False)

        self.cnv5 = nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=False)

        self.cnv6 = nn.Conv2d(1024, 2048, kernel_size=4, stride=2, padding=1, bias=False)

        self.leaky = nn.LeakyReLU(0.01, inplace=True)

        self.discrimination = nn.Conv2d(2048, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.classifier = nn.Conv2d(2048, dimensions, kernel_size=2, bias=False)

    def forward(self, input):
        layer = self.cnv1(input)
        layer = self.leaky(layer)

        layer = self.cnv2(layer)
        layer = self.leaky(layer)

        layer = self.cnv3(layer)
        layer = self.leaky(layer)

        layer = self.cnv4(layer)
        layer = self.leaky(layer)

        layer = self.cnv5(layer)
        layer = self.leaky(layer)

        layer = self.cnv6(layer)
        layer = self.leaky(layer)

        discrimination = self.discrimination(layer)
        classifications = self.classifier(layer)
        classifications = classifications.view(classifications.size(0), classifications.size(1))

        return discrimination, classifications


#Model architecture inspired by STARGan implementation https://github.com/yunjey/StarGAN

class ImageNetGenerator(nn.Module):

    def __init__(self, dimensions, image_size=128):
        super(ImageNetGenerator, self).__init__()

        self.image_size = image_size
        self.dimensions = dimensions

        self.cnv1 = nn.Conv2d(dimensions, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.norm64 = nn.InstanceNorm2d(64, affine=True)

        self.cnv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.norm128 = nn.InstanceNorm2d(128, affine=True)

        self.cnv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.norm256 = nn.InstanceNorm2d(256, affine=True)

        self.res1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)

        self.cnv4 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)

        self.cnv5 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)

        self.cnv6 = nn.ConvTranspose2d(64, 3, kernel_size=7, stride=1, padding=3, bias=False)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(True)

    def forward(self, input):
        layer = self.cnv1(input)
        layer = self.norm64(layer)
        layer = self.relu(layer)
        layer = self.cnv2(layer)
        layer = self.norm128(layer)
        layer = self.relu(layer)
        layer = self.cnv3(layer)
        layer = self.norm256(layer)
        layer = self.relu(layer)

        for i in range(6):
            layer = self.res1(layer)
            layer = self.norm256(layer)
            layer = self.relu(layer)
            layer = self.res1(layer)
            layer = self.norm256(layer)

        layer = self.cnv4(layer)
        layer = self.norm128(layer)
        layer = self.relu(layer)
        layer = self.cnv5(layer)
        layer = self.norm64(layer)
        layer = self.relu(layer)
        layer = self.cnv6(layer)
        output = self.tanh(layer)

        return output
