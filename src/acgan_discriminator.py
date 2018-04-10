
import os
import warnings
from torch import load, save
import torch.nn as nn

class ACDiscriminator(nn.Module):

    def __init__(self, num_classes):
        super(ACDiscriminator, self).__init__()

        self.num_classes = num_classes

        self.cnv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)

        self.cnv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0, bias=False)
        self.norm32 = nn.BatchNorm2d(32)

        self.cnv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm64 = nn.BatchNorm2d(64)

        self.cnv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0, bias=False)
        self.norm128 = nn.BatchNorm2d(128)

        self.cnv5 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm256 = nn.BatchNorm2d(256)

        self.cnv6 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0, bias=False)
        self.norm512 = nn.BatchNorm2d(512)

        self.leaky = nn.LeakyReLU(0.2, inplace=True)
        self.drop = nn.Dropout(0.5, inplace=False)

        self.fc_discrimination = nn.Linear(86528, 1)
        self.fc_prob_class = nn.Linear(86528, self.num_classes)

        self.softmax = nn.LogSoftmax(dim=0)
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

        flat6 = layer.view(-1, 86528)

        fc_discrimination = self.fc_discrimination(flat6)
        fc_prob_class = self.fc_prob_class(flat6)

        discrimination_results = self.sigmoid(fc_discrimination)
        class_results = self.softmax(fc_prob_class)

        return discrimination_results, class_results

    def Load(self, path):
        path = "./saves/" + path
        if not os.path.exists(path):
            raise ValueError("Saved model not provided, unable to load!")
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data = load(path)
                self.load_state_dict(data.state_dict())

    def Save(self, path):
        if not os.path.exists("./saves"):
            os.mkdir("./saves")

        save(self, path)
