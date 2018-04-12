import src.network as network
import torch.nn as nn

class Cifar10Discriminator(nn.Module):

    def __init__(self, num_classes):
        super(Cifar10Discriminator, self).__init__()

        self.num_classes = num_classes

        self.cnv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)

        self.cnv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm32 = nn.BatchNorm2d(32)

        self.cnv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm64 = nn.BatchNorm2d(64)

        self.cnv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm128 = nn.BatchNorm2d(128)

        self.cnv5 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm256 = nn.BatchNorm2d(256)

        self.cnv6 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm512 = nn.BatchNorm2d(512)

        self.leaky = nn.LeakyReLU(0.2, inplace=True)
        self.drop = nn.Dropout(0.5, inplace=False)

        self.fc_discrimination = nn.Linear(8192, 1)
        self.fc_prob_class = nn.Linear(8192, self.num_classes)

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        layer = self.cnv1(input)
        layer = self.leaky(layer)
        layer = self.drop(layer)

        layer = self.cnv2(layer)
        layer = self.norm32(layer)
        layer = self.leaky(layer)
        layer = self.drop(layer)

        layer = self.cnv3(layer)
        layer = self.norm64(layer)
        layer = self.leaky(layer)
        layer = self.drop(layer)

        layer = self.cnv4(layer)
        layer = self.norm128(layer)
        layer = self.leaky(layer)
        layer = self.drop(layer)

        layer = self.cnv5(layer)
        layer = self.norm256(layer)
        layer = self.leaky(layer)
        layer = self.drop(layer)

        layer = self.cnv6(layer)
        layer = self.norm512(layer)
        layer = self.leaky(layer)
        layer = self.drop(layer)

        flat6 = layer.view(-1, 8192)

        fc_discrimination = self.fc_discrimination(flat6)
        fc_prob_class = self.fc_prob_class(flat6)

        discrimination_results = self.sigmoid(fc_discrimination).view(-1,1).squeeze(1)
        class_results = self.softmax(fc_prob_class)

        return discrimination_results, class_results


class Cifar10Generator(nn.Module):

    def __init__(self,num_classes):
        super(Cifar10Generator, self).__init__()

        self.num_classes = num_classes

        self.fc1 = nn.Linear(110, 384)
        self.cnv1 = nn.ConvTranspose2d(384, 192, kernel_size=4, stride=1, padding=0, bias=False)
        self.norm192 = nn.BatchNorm2d(192)

        self.cnv2 = nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1, bias=False)
        self.norm96 = nn.BatchNorm2d(96)

        self.cnv3 = nn.ConvTranspose2d(96, 48, kernel_size=4, stride=2, padding=1, bias=False)
        self.norm48 = nn.BatchNorm2d(48)

        self.cnv4 = nn.ConvTranspose2d(48, 3, kernel_size=4, stride=2, padding=1, bias=False)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(True)

    def forward(self, input):
        input = input.view(-1, 110)
        fc = self.fc1(input)
        layer = fc.view(-1, 384, 1, 1)
        layer = self.cnv1(layer)
        layer = self.norm192(layer)
        layer = self.relu(layer)
        layer = self.cnv2(layer)
        layer = self.norm96(layer)
        layer = self.relu(layer)
        layer = self.cnv3(layer)
        layer = self.norm48(layer)
        layer = self.relu(layer)
        layer = self.cnv4(layer)
        output = self.tanh(layer)

        return output