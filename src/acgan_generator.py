
import torch.nn as nn


class ACGenerator(nn.Module):

    def __init__(self, image_size, ):
        super(ACGenerator, self).__init__()

        self.image_size = image_size
        self.fc1 = nn.Linear(110, 768)
        self.cnv1 = nn.ConvTranspose2d(768, 384, kernel_size=5, stride=2, padding=0, bias=False)
        self.norm512 = nn.BatchNorm2d(384)

        self.cnv2 = nn.ConvTranspose2d(384, 256, kernel_size=5, stride=2, padding=0, bias=False)
        self.norm256 = nn.BatchNorm2d(256)

        self.cnv3 = nn.ConvTranspose2d(256, 192, kernel_size=5, stride=2, padding=0, bias=False)
        self.norm192 = nn.BatchNorm2d(192)

        self.cnv4 = nn.ConvTranspose2d(192, self.image_size, kernel_size=5, stride=2, padding=0, bias=False)
        self.norm64 = nn.BatchNorm2d(64)

        self.cnv5 = nn.ConvTranspose2d(self.image_size, 3, kernel_size=8, stride=2, padding=0, bias=False)
        self.tanh = nn.Tanh()

        self.relu = nn.ReLU(True)

    def forward(self, input):
        input = input.view(-1, 110)
        fc = self.fc1(input)
        layer = fc.view(-1, 768, 1, 1)
        layer = self.cnv1(layer)
        layer = self.norm512(layer)
        layer = self.relu(layer)
        layer = self.cnv2(layer)
        layer = self.norm256(layer)
        layer = self.relu(layer)
        layer = self.cnv3(layer)
        layer = self.norm192(layer)
        layer = self.relu(layer)
        layer = self.cnv4(layer)
        layer = self.norm64(layer)
        layer = self.relu(layer)
        layer = self.cnv5(layer)
        output = self.tanh(layer)

        return output