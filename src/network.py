import os
import warnings
from torch import load, save

class Network:
    def __init__(self, config):

        if config.getboolean("train"):
            self.learning = float(config["learning"])
            self.momentum = float(config["momentum"])
            self.batch_size = int(config["batch"])
            self.epochs = int(config["epochs"])

        if config["saved_model"]:
            self.Load(config["saved_model"])

    def Load(self, path):
        path = "./saves/" + path
        if not os.path.exists(path):
            raise ValueError("Saved model not provided, unable to load cnet!")
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data = load(path)
                self.load_state_dict(data.state_dict())

    def Save(self, temp_epoch=None):
        if not os.path.exists("./saves"):
            os.mkdir("./saves")

        if temp_epoch is None:
            temp_epoch = self.epochs

        learning_string = str(self.learning)
        learning_string = learning_string[learning_string.find('.'):]

        filename = "cnet_epochs_{}_batch_{}_frames_{}_learning_{}.pt".format(temp_epoch, self.batch_size,
                                                                             self.frames, learning_string)

        save(self, "./saves/"+filename)
