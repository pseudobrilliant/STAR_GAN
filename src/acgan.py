import os
import torchvision.datasets as data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from src.imagenet_model import ImageNetGenerator,ImageNetDiscriminator
from src.cifar10_model import Cifar10Generator, Cifar10Discriminator
from src.network import BaseNetwork
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import numpy as np
import src.util as util
import torch.nn as nn
import warnings
from torch import load, save
import random

#Parts of this implementation were inspired by gitlimlabs ACGAN implimentation https://github.com/gitlimlab/ACGAN-PyTorch.
#Primarily the model structures, and the implementation of both Imagenet and CIFAR10 models

class ACGAN(BaseNetwork):

    def __init__(self, config):
        super(ACGAN, self).__init__(config)

        self.config = config
        if not os.path.exists("./saves"):
            os.makedirs("./saves")

        self.LoadDataset()

        self.LoadModels()

        self.train, self.val, self.test = util.split_dataset(self.dataset, self.batch_size)

        self.generator_optimizer = optim.Adam(self.generator.parameters(),
                                              lr=self.learning, betas=(0.5, 0.999))
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(),
                                                  lr=self.learning, betas=(0.5, 0.999))

        if self.is_cuda:
            self.discriminator = self.discriminator.cuda()
            self.generator = self.generator.cuda()

        self.Train()

    def LoadDataset(self):

        image_size = int(self.config["image_size"])

        self.dataset_type = self.config["dataset"]

        download = not os.path.exists("./dataset")

        if self.dataset_type == "imagenet":
            transform = transforms.Compose([
                transforms.CenterCrop(image_size * 2),
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

            self.dataset = data.ImageFolder(root='./dataset', transform=transform)
            self.classes = self.dataset.classes
            self.num_classes = len(self.dataset.classes)

        elif self.dataset_type == "cifar10":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

            self.classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            self.num_classes = 10
            self.dataset = data.CIFAR10(root='./dataset', transform=transform, download=download)
        else:
            print("ERROR: Unsupported Dataset Type")

    def LoadModels(self):

        if self.dataset_type == "imagenet":
            self.generator = ImageNetGenerator(self.num_classes)
            self.discriminator = ImageNetDiscriminator(self.num_classes)

        elif self.dataset_type == "cifar10":
            self.generator = Cifar10Generator(self.num_classes)
            self.discriminator = Cifar10Discriminator(self.num_classes)

        self.generator.apply(BaseNetwork.weights_init)
        self.discriminator.apply(BaseNetwork.weights_init)

        if "saved_generator" in self.config:
            self.generator = self.Load(self.config["saved_generator"], self.generator)

        if "saved_discriminator" in self.config:
            self.discriminator = self.Load(self.config["saved_discriminator"], self.discriminator)

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

                avg_discriminator_error += discriminator_error
                avg_generator_error += generator_error
                count += 1

            avg_discriminator_error /= count
            avg_generator_error /= count
            historical_discriminator_error.append(avg_discriminator_error)
            historical_generator_error.append(avg_generator_error)

            if self.verbose:
                print("\tAverage Discriminator Loss: {}".format(avg_discriminator_error))
                print("\tAverage Generator Loss: {}".format(avg_generator_error))

            val_dis_accuracy, val_discriminator_error, val_generator_error = self.Validate()
            historical_val_dis_accuracy.append(val_dis_accuracy)
            historical_val_discriminator_error.append(val_discriminator_error)
            historical_val_generator_error.append(val_generator_error)


            if self.report_period and (epoch + 1) % self.report_period == 0:
                self.StatusReport(epoch + 1, historical_discriminator_error, historical_generator_error,
                                  historical_val_dis_accuracy, historical_val_discriminator_error,
                                  historical_val_generator_error)

            if self.save_period and (epoch + 1) % self.save_period == 0:
                self.SaveModels(temp_epoch=(epoch+1))

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

        fake_size = 100 + self.num_classes
        fake_noise = np.random.normal(0, 1, (mini_batchsize, fake_size))

        if target_class is None:
            fake_labels = np.random.randint(0, self.num_classes, mini_batchsize)
        else:
            fake_labels = np.full(mini_batchsize, target_class)

        class_onehot = np.zeros((mini_batchsize, self.num_classes))
        class_onehot[np.arange(mini_batchsize), fake_labels] = 1
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

        dis_accuracy = (real_dis_accurate + fake_dis_accurate) / (len(real_var.data) * 2) * 100

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
        plt.ioff()
        fig, ax = plt.subplots(int(self.num_classes / 5), 5, figsize=(32, 32))
        fig.set_size_inches(6,3)

        for i in range(self.num_classes):
            fake_var, fake_dis_var, fake_class_var = self.GetFakeVariables(1, target_class=i)
            fake_var_cpu = fake_var.data[0]

            grid = vutils.make_grid(tensor=fake_var_cpu)
            ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            image = Image.fromarray(ndarr)
            row = int(i/5)
            col = int(i%5)
            ax[row, col].cla()
            ax[row, col].imshow(image, interpolation='none')
            ax[row, col].get_xaxis().set_visible(False)
            ax[row, col].get_yaxis().set_visible(False)
            ax[row, col].set_title(self.classes[i], fontsize=15)
            vutils.save_image(fake_var.data.cpu(), "./saves/{}.png".format(self.classes[i]))

        plt.tight_layout()
        plt.savefig(path, bbox_inches='tight', dpi=100)
        plt.close()

    def SaveModels(self, temp_epoch=None):

        if temp_epoch is None:
            temp_epoch = self.epochs

        learning_string = str(self.learning)
        learning_string = learning_string[learning_string.find('.'):]

        filename = "./saves/epochs_{}_batch_{}_learning_{}".format(temp_epoch, self.batch_size,
                                                                                    learning_string)
        self.Save(filename+"_discriminator.pt",self.discriminator)
        self.Save(filename+"_generator.pt", self.generator)



    def Load(self, path, obj):
        path = path
        if not os.path.exists(path):
            raise ValueError("Saved model not provided, unable to load!")
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data = load(path)
                obj.load_state_dict(data.state_dict())
        return obj


    def Save(self, path, obj):
        if not os.path.exists("./saves"):
            os.mkdir("./saves")

        obj = save(obj, path)

