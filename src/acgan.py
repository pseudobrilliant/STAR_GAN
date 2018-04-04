import os
import torch
import torchvision.transforms as transforms
from src.acgan_generator import ACGenerator
from src.acgan_discriminator import ACDiscriminator
from src.network import Network
import src.util as util
import torch.nn as nn
from torch import load, save
import warnings

class ACGAN:

    def __init__(self, config):
        super(Network, self, config).__init__()

        self.config = config
        self.generator = ACGenerator()
        self.discriminator = ACDiscriminator()

    def Train(self, dataset):

        train, val = util.validation_split(dataset, 0.15)

        trainloader = torch.utils.data.DataLoader(train, batch_size=self.batch_size, shuffle=True)

        discrimination_criterion = nn.BCELoss()
        classification_criterion = nn.NLLLoss()

        for epoch in range(super.epochs):
            for i, data in enumerate(trainloader, 0):
                dis_result, class_result = self.discriminator(input)

                discrimination_error = discrimination_criterion(dis_result, 1)
                aux_errD_real = classification_criterion(class_result, )
                errD_real = dis_errD_real + aux_errD_real
                errD_real.backward()







    def Validation(self, dataset):

        valloader = torch.utils.data.DataLoader(val, batch_size=self.batch_size, shuffle=True)




    def TrainRealSample(self, data):
        self.discriminator.zero_grad()
        image,label = data

        image = image.cuda()



    def TrainFakeSample(self):

    def GenerateFakeSample(self):