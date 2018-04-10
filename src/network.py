import os
import warnings
from torch import load, save
import torch.cuda

class BaseNetwork:
    def __init__(self, config):

        self.is_cuda = torch.cuda.is_available()

        if config.getboolean("train"):
            self.learning = float(config["learning"])
            self.momentum = float(config["momentum"])
            self.batch_size = int(config["batch"])
            self.epochs = int(config["epochs"])
            self.verbose = config.getboolean("verbose")
            self.report_period = int(config["report_period"])
            self.save_period = int(config["save_period"])

        if "saved_model" in config:
            self.Load(config["saved_model"])

