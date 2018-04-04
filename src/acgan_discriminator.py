import torch.nn as nn


class ACDiscriminator(nn.Module):

    def __init__(self, classifications):
        super(ACDiscriminator, self).__init__()

        self.cnv1 = nn.ConvTranspose2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)

        self.cnv2 = nn.ConvTranspose2d(16, 32, kernel_size=3, stride=1, padding=0, bias=False)
        self.norm32 = nn.BatchNorm2d(32)

        self.cnv3 = nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm64 = nn.BatchNorm2d(64)

        self.cnv4 = nn.ConvTranspose2d(64, 128, kernel_size=3, stride=1, padding=0, bias=False)
        self.norm128 = nn.BatchNorm2d(128)

        self.cnv5 = nn.ConvTranspose2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm256 = nn.BatchNorm2d(256)

        self.cnv6 = nn.ConvTranspose2d(256, 512, kernel_size=3, stride=1, padding=0, bias=False)
        self.norm512 = nn.BatchNorm2d(512)

        self.leaky = nn.LeakyReLU(0.2, inplace=True)
        self.drop = nn.Dropout(0.5, inplace=False)

        self.fc_discrimination = nn.Linear(86528, 1)
        self.fc_prob_class = nn.Linear(86528, classifications)

        self.softmax = nn.Softmax()
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

        flat6 = layer.view(-1, 13 * 13 * 512)

        fc_discrimination = self.fc_discrimination(flat6)
        fc_prob_class = self.fc_prob_class(flat6)

        class_results = self.softmax(fc_prob_class)
        discrimination_results = self.softmax(fc_discrimination)

        return class_results, discrimination_results

