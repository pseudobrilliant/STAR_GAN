import os
import torchvision.datasets as data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from src.imagenet_model import ImageNetGenerator, ImageNetDiscriminator
from src.cifar10_model import Cifar10Generator, Cifar10Discriminator
from src.network import BaseNetwork
import src.util as utils
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


# Parts of this implementation were inspired by gitlimlabs ACGAN implimentation https://github.com/gitlimlab/ACGAN-PyTorch.
# Primarily the model structures, and the implementation of both Imagenet and CIFAR10 models

class ACGAN(BaseNetwork):

    def __init__(self, config):
        super(ACGAN, self).__init__(config)

        self.config = config
        if not os.path.exists("./saves"):
            os.makedirs("./saves")

        self.LoadDatasets()

        self.LoadModels()

        self.LoadTraining()

    def LoadDatasets(self):

        self.image_size = int(self.config["image_size"])

        self.dataset_types = self.config["dataset"].split(',')

        self.datasets = []

        self.total_num_classes = 0

        temp_dataset = temp_classes = temp_num_classes = None

        for dataset in self.dataset_types:
            if dataset == "imagenet":
                if not os.path.exists("./imagenet"):
                    utils.download_dataset("imagenet", "http://pseudobrilliant.com/files/imagenet.zip", "./imagenet")

                transform = transforms.Compose([
                    transforms.CenterCrop(self.image_size * 2),
                    transforms.Resize(self.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

                temp_dataset = data.ImageFolder(root='./imagenet', transform=transform)
                temp_classes = temp_dataset.classes
                temp_num_classes = len(temp_dataset.classes)

            elif dataset == "cifar10":
                download = not os.path.exists("./cifar10")

                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

                temp_classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
                temp_num_classes = 10
                temp_dataset = data.CIFAR10(root='./cifar10', transform=transform, download=download)

            elif dataset == "wikiart":

                if not os.path.exists("./wikiart"):
                    utils.download_dataset("wikiart", "http://pseudobrilliant.com/files/wikiart.zip", "./wikiart")

                transform = transforms.Compose([
                    transforms.CenterCrop(self.image_size * 4),
                    transforms.Resize(self.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

                temp_dataset = data.ImageFolder(root='./wikiart', transform=transform)
                temp_classes = temp_dataset.classes
                temp_num_classes = len(temp_dataset.classes)
            else:
                print("ERROR: Unsupported Dataset Type")
                exit(0)

            training, val, test = util.split_dataset(temp_dataset, self.batch_size, val_split=0.10,test_split=0.5)

            self.datasets.append({"type": dataset, "data": temp_dataset, "training": training, "val": val, "test": test,
                                  "classes": temp_classes, "num_classes": temp_num_classes})

            self.total_num_classes += temp_num_classes

        self.num_datasets = len(self.datasets)

    def LoadModels(self):

        self.dimensions = self.total_num_classes + self.num_datasets + 3;

        if self.image_size > 32:
            self.generator = ImageNetGenerator(self.dimensions)
            self.discriminator = ImageNetDiscriminator(self.total_num_classes)
        else:
            self.generator = Cifar10Generator(self.dimensions)
            self.discriminator = Cifar10Discriminator(self.total_num_classes)

        if self.is_cuda:
            self.discriminator = self.discriminator.cuda()
            self.generator = self.generator.cuda()

        self.generator.apply(BaseNetwork.weights_init)
        self.discriminator.apply(BaseNetwork.weights_init)

        if "saved_generator" in self.config:
            self.generator = self.Load(self.config["saved_generator"], self.generator)

        if "saved_discriminator" in self.config:
            self.discriminator = self.Load(self.config["saved_discriminator"], self.discriminator)

    def LoadTraining(self):
        if self.train:

            self.generator_optimizer = optim.Adam(self.generator.parameters(),
                                                  lr=self.learning, betas=(0.5, 0.999))
            self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(),
                                                      lr=self.learning, betas=(0.5, 0.999))

            self.critic_iterations = int(self.config['critic_iterations'])

            self.discrimination_criterion = nn.BCELoss()
            self.classification_criterion = nn.CrossEntropyLoss()

            if self.is_cuda:
                self.discrimination_criterion = self.discrimination_criterion.cuda()
                self.classification_criterion = self.classification_criterion.cuda()

            self.Train()

    def Train(self):

        historical_discriminator_error = []
        historical_generator_error = []
        historical_discriminator_real = []
        historical_discriminator_fake = []
        historical_discriminator_accuracy = []
        historical_discrimination_loss = []
        total_discriminator_error=0
        total_generator_error = 0

        if self.early_training is not None:
            self.EarlyTraining()

        for epoch in range(self.epochs):

            if self.verbose:
                print("Epoch {}:".format(epoch))

            discriminator_error = self.TrainDiscriminator()
            generator_error = self.TrainGenerator()
            total_discriminator_error += discriminator_error
            total_generator_error += generator_error

            if self.verbose:
                print("\tAverage Discriminator Loss: {}".format(discriminator_error))
                print("\tAverage Generator Loss: {}".format(generator_error))

            if self.report_period and (epoch + 1) % self.report_period == 0:
                historical_discriminator_error.append(total_discriminator_error / (epoch+1))
                historical_generator_error.append(total_generator_error / (epoch+1))
                real_acc, fake_acc, total_acc, disc_loss = self.Validation()
                historical_discriminator_real.append(real_acc)
                historical_discriminator_fake.append(fake_acc)
                historical_discriminator_accuracy.append(total_acc)
                historical_discrimination_loss.append(disc_loss)
                self.StatusReport(epoch + 1, historical_discriminator_error,
                                  historical_generator_error,
                                  historical_discriminator_real,
                                  historical_discriminator_fake,
                                  historical_discriminator_accuracy,
                                  historical_discrimination_loss)

            if self.save_period and (epoch + 1) % self.save_period == 0:
                self.SaveModels(temp_epoch=(epoch + 1))

    def GetRealVariables(self, batch, labels):

        mini_batchsize = len(batch)

        real_var = Variable(batch)

        real_dis_val = torch.ones(mini_batchsize)
        real_dis_var = Variable(real_dis_val)

        real_class_var = Variable(labels)

        if self.is_cuda:
            real_var = real_var.cuda()
            real_dis_var = real_dis_var.cuda()
            real_class_var = real_class_var.cuda()

        return real_var, real_dis_var, real_class_var

    def GetFakeVariables(self, dataset_id, batch, target_class=None, full_labels=None):

        mini_batchsize = len(batch)

        mask_size = self.total_num_classes + self.num_datasets
        fake_size = self.image_size + mask_size

        mask = None

        if full_labels is None:
            for i in range(len(self.datasets)):

                class_onehot = np.zeros((mini_batchsize, self.datasets[i]["num_classes"]))

                if dataset_id == i and target_class is None:
                    fake_labels = np.random.randint(0, self.datasets[i]["num_classes"], mini_batchsize)
                    class_onehot[np.arange(mini_batchsize), fake_labels] = 1
                    fake_labels_val = torch.from_numpy(fake_labels)
                elif dataset_id == i and target_class is not None:
                    fake_labels = target_class
                    class_onehot[np.arange(mini_batchsize), target_class] = 1
                    fake_labels_val = fake_labels

                if mask is None:
                    mask = class_onehot
                else:
                    mask = np.concatenate((mask, class_onehot), axis=1)
        else:
            mask = np.zeros(mini_batchsize, mask_size)
            mask = mask[np.arange(mini_batchsize)] = full_labels

        start = 0
        for i in range(len(self.datasets)):
            if dataset_id == i or (
                    full_labels is not None and 1 in full_labels[start:start + self.datasets[i]["num_classes"]]):
                flags = np.ones((mini_batchsize, 1))
            else:
                flags = np.zeros((mini_batchsize, 1))

            mask = np.concatenate((mask, flags), axis=1)
            start = self.datasets[i]["num_classes"]

        # Turns the one hot vector into channels. If the 0 onehot class was 1 then the entire 4th channel will be ones.
        # This creates an image of batch_size * 3 + num_classes * w *h
        mask_torch = torch.from_numpy(mask)
        mask_torch = mask_torch.view(mask_torch.size(0), mask_torch.size(1), 1, 1)
        mask_torch = mask_torch.repeat(1, 1, batch.size(2), batch.size(3)).float()
        labeled_batch = torch.cat([batch, mask_torch], dim=1)

        labeled_batch_var = Variable(labeled_batch)

        fake_dis_val = torch.zeros(mini_batchsize)
        fake_dis_var = Variable(fake_dis_val)

        fake_class_val = fake_labels_val.long()
        fake_class_var = Variable(fake_class_val)

        if self.is_cuda:
            labeled_batch_var = labeled_batch_var.cuda()
            fake_dis_var = fake_dis_var.cuda()
            fake_class_var = fake_class_var.cuda()

        fake_var = self.generator(labeled_batch_var)

        return fake_var, fake_dis_var, fake_class_var

    def EarlyTraining(self):

        print("Running Pretraining")

        dataset_class_start = 0
        for i in range(self.num_datasets):
            for j in range(self.early_training):
                self.discriminator_optimizer.zero_grad()

                data = iter(self.datasets[i]["training"])
                dataset_class_end = self.datasets[i]["num_classes"] + dataset_class_start
                batch, label = data.next()

                (real_var, real_dis_var, real_class_var) = self.GetRealVariables(batch, label)

                real_dis_result, real_class_result = self.discriminator(real_var)

                real_class_result_split = real_class_result[:, dataset_class_start:dataset_class_end]
                real_classification_error = self.classification_criterion(real_class_result_split, real_class_var)

                real_classification_error.backward()

                self.discriminator_optimizer.step()

            dataset_class_start = dataset_class_end

    def TrainDiscriminator(self):

        total_discriminator_error_cpu = 0

        for i in range(self.critic_iterations):
            dataset_class_start = 0
            for i in range(len(self.datasets)):

                self.discriminator.zero_grad()

                data = iter(self.datasets[i]["training"])
                dataset_class_end = self.datasets[i]["num_classes"] + dataset_class_start
                batch, label = data.next()

                (real_var, real_dis_var, real_class_var) = self.GetRealVariables(batch, label)
                (fake_var, fake_dis_var, fake_class_var) = self.GetFakeVariables(i, batch)

                real_dis_result, real_class_result = self.discriminator(real_var)

                real_discrimination_error = - torch.mean(real_dis_result)

                real_class_result_split = real_class_result[:, dataset_class_start:dataset_class_end]
                real_classification_error = self.classification_criterion(real_class_result_split, real_class_var)

                detach_fake = fake_var.clone().detach()
                fake_dis_result, fake_class_result = self.discriminator(detach_fake)

                fake_discrimination_error = torch.mean(fake_dis_result)

                # Compute loss for gradient penalty.
                alpha = torch.rand(real_var.size(0), 1, 1, 1)
                if self.is_cuda:
                    alpha = alpha.cuda()

                # Gradient Penalty https://github.com/yunjey/StarGAN
                x_hat = Variable(alpha * real_var.data + (1 - alpha) * fake_var.data, requires_grad=True)
                out_src, _ = self.discriminator(x_hat)
                d_loss_gp = self.GradientPenalty(out_src, x_hat)

                loss = real_discrimination_error + fake_discrimination_error + \
                       1 * real_classification_error + \
                       10 * d_loss_gp

                loss.backward()
                total_discriminator_error_cpu += loss.data.cpu().numpy()[0]
                self.discriminator_optimizer.step()

                dataset_class_start = dataset_class_end

        return total_discriminator_error_cpu / (self.critic_iterations * self.num_datasets)

    def TrainGenerator(self):
        total_generator_error_cpu = 0
        dataset_class_start = 0
        for i in range(len(self.datasets)):
            self.generator.zero_grad()

            data = iter(self.datasets[i]["training"])
            batch, labels = data.next()

            dataset_class_end = dataset_class_start + self.datasets[i]["num_classes"]

            (real_var, real_dis_var, real_class_var) = self.GetRealVariables(batch, labels)
            (fake_var, fake_dis_var, fake_class_var) = self.GetFakeVariables(i, batch)

            fake_dis_result, fake_class_result = self.discriminator(fake_var)

            generator_discrimination_error = - torch.mean(fake_dis_result)

            fake_class_result_split = fake_class_result[:, dataset_class_start:dataset_class_end]
            generator_classification_error = self.classification_criterion(fake_class_result_split, fake_class_var)

            fake_tensor = fake_var.data.cpu()

            (recnst_var, recnst_dis_var, recnst_class_var) = self.GetFakeVariables(i, fake_tensor, labels)

            reconstruct_loss = torch.mean(torch.abs(real_var - recnst_var))

            loss = generator_discrimination_error + 10 * reconstruct_loss + 1 * generator_classification_error
            total_generator_error_cpu += loss.data.cpu().numpy()[0]

            loss.backward()
            self.generator_optimizer.step()

            dataset_class_start = dataset_class_end

        return total_generator_error_cpu / self.num_datasets

    # Gradient Penalty by STARGan implementation https://github.com/yunjey/StarGAN
    def GradientPenalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size())

        if self.is_cuda:
            weight = weight.cuda()

        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))

        return torch.mean((dydx_l2norm - 1) ** 2)

    def Validation(self):

        real_right = 0
        fake_right = 0
        total_count = 0
        dataset_class_start = 0
        generator_discrimination_error = 0

        for i in range(self.num_datasets):
            dataset = self.datasets[i]["val"]

            for batch, labels in dataset:
                dataset_class_end = dataset_class_start + self.datasets[i]["num_classes"]

                (real_var, real_dis_var, real_class_var) = self.GetRealVariables(batch, labels)
                (fake_var, fake_dis_var, fake_class_var) = self.GetFakeVariables(i, batch, labels)

                real_dis_result, real_class_result = self.discriminator(real_var)
                real_class_result_split = real_class_result[:, dataset_class_start:dataset_class_end]
                real_right += self.GetAccuracy(real_class_result_split, real_class_var)

                generator_discrimination_error += - torch.mean(real_dis_result).data.cpu().numpy()[0]

                fake_dis_result, fake_class_result = self.discriminator(fake_var)
                fake_class_result_split = fake_class_result[:, dataset_class_start:dataset_class_end]
                fake_right += self.GetAccuracy(fake_class_result_split, fake_class_var)

                generator_discrimination_error += - torch.mean(fake_dis_result).data.cpu().numpy()[0]

                total_count += len(batch)

            dataset_class_start = dataset_class_end

        return (real_right / (total_count * 2)) * 100.0, \
               (fake_right / (total_count * 2)) * 100.0, \
               ((real_right + fake_right) / (total_count * 4)) * 100.0, \
               generator_discrimination_error / (total_count * 4)

    def GetAccuracy(self, source, target):

        source_cpu = source.data.cpu().numpy()
        target_cpu = target.data.cpu().numpy()
        num_correct = 0
        for i in range(len(source_cpu)):
            if np.argmax(source_cpu[i]) == target_cpu[i]:
                num_correct += 1

        return num_correct

    def StatusReport(self, epoch, historical_discriminator_error,
                     historical_generator_error,
                     historical_discriminator_real,
                     historical_discriminator_fake,
                     historical_discriminator_accuracy,
                     historical_discrimination_loss):

        x = [(i + 1) * 100 for i in range(0, int((epoch+1) / self.report_period))]
        plt.title("Iteration vs Average Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Average Loss")
        plt.plot(x, historical_discriminator_error, marker='o', label='discriminator loss')
        plt.plot(x, historical_generator_error, marker='o', label='generator loss')
        plt.legend(loc='best')
        plt.savefig('./saves/epoch_{}_loss.png'.format(epoch))

        plt.close()

        plt.title("Iteration vs Accuracy")
        plt.xlabel("Iteration")
        plt.ylabel("Average Accuracy")
        plt.plot(x, historical_discriminator_real, marker='o', label='class real')
        plt.plot(x, historical_discriminator_fake, marker='o', label='class fake')
        plt.plot(x, historical_discriminator_accuracy, marker='o', label='class total')
        plt.legend(loc='best')
        plt.savefig('./saves/epoch_{}_accuracy.png'.format(epoch))

        plt.close()

        plt.title("Iteration vs Discriminator Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Average Discriminator Loss")
        plt.plot(x, historical_discriminator_real, marker='o', label='discrimination only loss')
        plt.legend(loc='best')
        plt.savefig('./saves/epoch_{}_disc_loss.png'.format(epoch))

        plt.close()

        self.GenerateSampleImages("./saves/training_samples_{}_epochs".format(epoch))

    # Visuzlization method https://github.com/yunjey/StarGAN
    def GenerateSampleImages(self, path='./saves/img.png'):
        data = iter(self.datasets[0]["test"])
        batch, labels = data.next()

        fake = [batch]
        for i in range(self.datasets[0]["num_classes"]):
            trg = torch.from_numpy(np.repeat(i, len(batch)))
            fake_var, fake_dis_var, fake_class_var = self.GetFakeVariables(0, batch, trg)
            fake.append(fake_var.data.cpu())
        for i in range(self.datasets[1]["num_classes"]):
            trg = torch.from_numpy(np.repeat(i, len(batch)))
            fake_var, fake_dis_var, fake_class_var = self.GetFakeVariables(1, batch, trg)
            fake.append(fake_var.data.cpu())

        image = torch.cat(fake, dim=3)
        out = (image + 1) / 2
        out.clamp_(0, 1)

        vutils.save_image(out, "{}_current_samples_{}_fake.png".format(path, self.datasets[0]["type"]), nrow=1)

    def SaveModels(self, temp_epoch=None):

        if temp_epoch is None:
            temp_epoch = self.epochs

        learning_string = str(self.learning)
        learning_string = learning_string[learning_string.find('.'):]

        filename = "./saves/epochs_{}_batch_{}_learning_{}".format(temp_epoch, self.batch_size,
                                                                   learning_string)
        self.Save(filename + "_discriminator.pt", self.discriminator)
        self.Save(filename + "_generator.pt", self.generator)

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
