import os
import torchvision.datasets as data
import torchvision.utils as vutils
from torch.autograd import Variable
from src.acgan_generator import ACGenerator
from src.acgan_discriminator import ACDiscriminator
from src.network import BaseNetwork
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import numpy as np
import src.util as util
import torch.nn as nn
import random

class ACGAN(BaseNetwork):

    def __init__(self, config, transform):
        super(ACGAN, self).__init__(config)

        self.config = config
        if not os.path.exists("./saves"):
            os.makedirs("./saves")

        self.dataset = data.ImageFolder(root='./dataset', transform=transform)
        self.classes = self.dataset.classes
        self.num_classes = len(self.dataset.classes)
        self.train, self.val, self.test = util.split_dataset(self.dataset, self.batch_size)

        self.generator = ACGenerator(self.num_classes)
        self.discriminator = ACDiscriminator(self.num_classes)

        self.generator_optimizer = optim.Adam(self.generator.parameters(),
                                              lr=self.learning, betas=(0.5, 0.999))
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(),
                                                  lr=self.learning, betas=(0.5, 0.999))

        if self.is_cuda:
            self.discriminator = self.discriminator.cuda()
            self.generator = self.generator.cuda()

        self.Train()

    def Train(self):

        historical_discriminator_error = []
        historical_generator_error = []
        historical_val_dis_accuracy = []
        historical_val_discriminator_error = []
        historical_val_generator_error = []

        for epoch in range(self.epochs):

            if self.verbose:
                print("Epoch {}:".format(epoch))

            avg_discriminator_error = 0
            avg_generator_error = 0
            count = 0

            for i, data in enumerate(self.train, 0):
                real_variables = self.GetRealVariables(data)
                fake_variables = self.GetFakeVariables(len(data[0]))

                discriminator_error, generator_error = self.TrainSamples(real_variables, fake_variables)

                del real_variables, fake_variables
                torch.cuda.empty_cache()

                avg_discriminator_error += discriminator_error
                avg_generator_error += generator_error
                count += 1

            avg_discriminator_error /= count
            avg_generator_error /= count
            historical_discriminator_error.append(avg_discriminator_error)
            historical_generator_error.append(avg_generator_error)

            if self.verbose:
                print("\tAverage Discriminator Accuracy: {}".format(avg_discriminator_error))
                print("\tAverage Generator Accuracy: {}".format(avg_generator_error))

            val_dis_accuracy, val_discriminator_error, val_generator_error = self.Validate()
            historical_val_dis_accuracy.append(val_dis_accuracy)
            historical_val_discriminator_error.append(val_discriminator_error)
            historical_val_generator_error.append(val_generator_error)


            if self.report_period and (epoch + 1) % self.report_period == 0:
                self.StatusReport(epoch + 1, historical_discriminator_error, historical_generator_error,
                                  historical_val_dis_accuracy, historical_val_discriminator_error,
                                  historical_val_generator_error)

            if self.save_period and (epoch + 1) % self.save_period == 0:
                self.Save(temp_epoch=(epoch+1))

    def GetRealVariables(self, data):

        batch, labels = data
        mini_batchsize = len(batch)

        real_var = Variable(batch)

        real_dis_val = torch.ones(mini_batchsize)
        real_dis_var = Variable(real_dis_val).unsqueeze(1)

        real_class_var = Variable(labels)

        if self.is_cuda:
            real_var = real_var.cuda()
            real_dis_var = real_dis_var.cuda()
            real_class_var = real_class_var.cuda()

        return real_var, real_dis_var, real_class_var

    def GetFakeVariables(self, mini_batchsize, target_class=None):

        fake_noise = np.random.normal(0, 1, (mini_batchsize, 110))

        if target_class is None:
            fake_labels = np.random.randint(0, self.num_classes, mini_batchsize)
        else:
            fake_labels = np.full(mini_batchsize, target_class)

        class_onehot = np.zeros((mini_batchsize, self.num_classes))
        class_onehot[0:mini_batchsize, fake_labels] = 1
        fake_noise[0:mini_batchsize, :self.num_classes] = class_onehot

        fake_noise = (torch.from_numpy(fake_noise)).float()
        fake_noise_var = Variable(fake_noise)

        fake_dis_val = torch.zeros(mini_batchsize)
        fake_dis_var = Variable(fake_dis_val).unsqueeze(1)

        fake_class_val = torch.from_numpy(fake_labels).long()
        fake_class_var = Variable(fake_class_val)

        if self.is_cuda:
            fake_noise_var = fake_noise_var.cuda()
            fake_dis_var = fake_dis_var.cuda()
            fake_class_var = fake_class_var.cuda()

        fake_var = self.generator(fake_noise_var)

        return fake_var, fake_dis_var, fake_class_var

    def TrainSamples(self, real_variables, fake_variables):

        self.discriminator.zero_grad()

        discrimination_criterion = nn.BCELoss()
        classification_criterion = nn.NLLLoss()

        (real_var, real_dis_var, real_class_var) = real_variables
        (fake_var, fake_dis_var, fake_class_var) = fake_variables

        real_dis_result, real_class_result = self.discriminator(real_var)

        real_discrimination_error = discrimination_criterion(real_dis_result, real_dis_var)
        real_classification_error = classification_criterion(real_class_result, real_class_var)
        real_discriminator_error = real_discrimination_error + real_classification_error
        real_discriminator_error.backward()

        fake_dis_result, fake_class_result = self.discriminator(fake_var.detach())

        fake_discrimination_error = discrimination_criterion(fake_dis_result, fake_dis_var)
        fake_classification_error = classification_criterion(fake_class_result, fake_class_var)
        fake_discriminator_error = fake_discrimination_error + fake_classification_error
        fake_discriminator_error.backward()

        total_discriminator_error_cpu = (fake_discriminator_error + real_discriminator_error).data.cpu().numpy()[0]

        self.discriminator_optimizer.step()
        self.generator.zero_grad()

        fake_dis_result, fake_class_result = self.discriminator(fake_var)

        generator_discrimination_error = discrimination_criterion(fake_dis_result, real_dis_var)
        generator_classification_error = classification_criterion(fake_class_result, fake_class_var)
        total_generator_error = generator_discrimination_error + generator_classification_error
        total_generator_error.backward()

        total_generator_error_cpu = total_generator_error.data.cpu().numpy()[0]

        self.generator_optimizer.step()

        del discrimination_criterion, classification_criterion, \
            real_discriminator_error, real_classification_error, real_discrimination_error, \
            fake_discriminator_error, fake_classification_error, fake_discrimination_error
        torch.cuda.empty_cache()

        return total_discriminator_error_cpu, total_generator_error_cpu

    def Validate(self):

        avg_dis_accuracy = 0
        avg_discriminator_error = 0
        avg_generator_error = 0
        count = 0

        for i, data in enumerate(self.val, 0):
            real_variables = self.GetRealVariables(data)
            fake_variables = self.GetFakeVariables(len(data[0]))

            dis_accuracy, discriminator_error, generator_error = self.ValidateSamples(real_variables, fake_variables)
            avg_dis_accuracy += dis_accuracy
            avg_discriminator_error += discriminator_error
            avg_generator_error += generator_error
            count += 1

            del real_variables, fake_variables
            torch.cuda.empty_cache()

        avg_dis_accuracy /= count
        avg_discriminator_error /= count
        avg_generator_error /= count

        if self.verbose:
            print("\tValidation Discriminator Accuracy: {}".format(avg_dis_accuracy))
            print("\tValidation Discriminator Error: {}".format(avg_discriminator_error))
            print("\tValidation Generator Error: {}".format(avg_generator_error))

        return avg_dis_accuracy, avg_discriminator_error, avg_generator_error

    def ValidateSamples(self, real_variables, fake_variables):

        discrimination_criterion = nn.BCELoss()
        classification_criterion = nn.NLLLoss()

        (real_var, real_dis_var, real_class_var) = real_variables
        (fake_var, fake_dis_var, fake_class_var) = fake_variables

        real_dis_result, real_class_result = self.discriminator(real_var)

        real_dis_accurate, real_dis_accurate_perc = self.GetDiscriminationAccuracy(real_dis_result, real_dis_var)
        real_discrimination_error = discrimination_criterion(real_dis_result, real_dis_var)
        real_classification_error = classification_criterion(real_class_result, real_class_var)
        real_discriminator_error = real_discrimination_error + real_classification_error

        fake_dis_result, fake_class_result = self.discriminator(fake_var.detach())

        fake_dis_accurate, fake_dis_accurate_perc = self.GetDiscriminationAccuracy(fake_dis_result, fake_dis_var)
        fake_discrimination_error = discrimination_criterion(fake_dis_result, fake_dis_var)
        fake_classification_error = classification_criterion(fake_class_result, fake_class_var)
        fake_discriminator_error = fake_discrimination_error + fake_classification_error

        dis_accuracy = (real_dis_accurate_perc + fake_dis_accurate_perc) / (len(real_var.data) * 2) * 100

        total_discriminator_error_cpu = (fake_discriminator_error + real_discriminator_error).data.cpu().numpy()[0]

        generator_discrimination_error = discrimination_criterion(fake_dis_result, real_dis_var)
        generator_classification_error = classification_criterion(fake_class_result, fake_class_var)
        total_generator_error = generator_discrimination_error + generator_classification_error

        total_generator_error_cpu = total_generator_error.data.cpu().numpy()[0]

        del discrimination_criterion, classification_criterion, \
            real_discriminator_error, real_classification_error, real_discrimination_error, \
            fake_discriminator_error, fake_classification_error, fake_discrimination_error
        torch.cuda.empty_cache()

        return dis_accuracy, total_discriminator_error_cpu, total_generator_error_cpu

    def GetDiscriminationAccuracy(self, source, target):
        source_cpu = source.data.cpu().numpy()
        target_cpu = target.data.cpu().numpy()
        num_correct = 0
        for i in range(len(source_cpu)):
            if source_cpu[i] > 0.85 and target_cpu[i] == 1:
                num_correct += 1

        percent_correct = (num_correct / len(source_cpu)) * 100
        return num_correct, percent_correct

    def StatusReport(self, epoch, historical_discriminator_error, historical_generator_error,
                     historical_val_dis_accuracy, historical_val_discriminator_error,
                     historical_val_generator_error):

        x = [i for i in range(1, epoch + 1)]
        plt.title("Epoch vs Discriminator Average Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Average Loss")
        plt.plot(x, historical_discriminator_error, marker='o', label='training loss')
        plt.plot(x, historical_val_discriminator_error, marker='o', label='val loss')
        plt.legend(loc='best')
        plt.savefig('./saves/epoch_{}_disc.png'.format(epoch))

        plt.close()

        plt.title("Epoch vs Generator Average Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Average Loss")
        plt.plot(x, historical_generator_error, marker='o', label='training loss')
        plt.plot(x, historical_val_generator_error, marker='o', label='val loss')
        plt.legend(loc='best')
        plt.savefig('./saves/epoch_{}_generator.png'.format(epoch))

        plt.close()

        plt.title("Epoch vs Discriminator Average Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Average Accuracy")
        plt.plot(x, historical_val_dis_accuracy, marker='o')
        plt.savefig('./saves/epoch_{}_accuracy.png'.format(epoch))

        plt.close()

        self.GenerateSampleImages("./saves/training_samples_{}_epochs.png".format(epoch))

    def GenerateSampleImages(self, path='./saves/img.png'):

        fig = plt.figure(figsize=(64, 64))
        for i in range(self.num_classes):
            fake_var, fake_dis_var, fake_class_var = self.GetFakeVariables(1, target_class=i)
            fake_var_cpu = fake_var.data.cpu()
            fake_var_cpu = torch.clamp(fake_var_cpu, min=0.0, max=1.0).cpu()[0]
            image = np.transpose(fake_var_cpu, (2, 1, 0))
            ax = fig.add_subplot(5, 2, i+1)
            ax.imshow(image)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_title(self.classes[i], size=60)

        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def Save(self, temp_epoch=None):

        if temp_epoch is None:
            temp_epoch = self.epochs

        learning_string = str(self.learning)
        learning_string = learning_string[learning_string.find('.'):]

        filename = "./saves/epochs_{}_batch_{}_learning_{}".format(temp_epoch, self.batch_size,
                                                                                    learning_string)

        self.discriminator.Save(filename+"_discriminator.pt")
        self.generator.Save(filename+"_generator.pt")

    def Load(self, path):
        self.discriminator.Load(path)
        self.generator.Load(path)




