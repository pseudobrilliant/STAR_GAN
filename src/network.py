import random
import torch.cuda


class BaseNetwork:
    def __init__(self, config):

        seed = random.randint(1, 10000)
        random.seed(seed)

        self.is_cuda = torch.cuda.is_available()

        self.batch_size = int(config["batch"])
        self.train = config.getboolean("train")
        if self.train:
            self.learning = float(config["learning"])
            self.epochs = int(config["epochs"])
            self.verbose = config.getboolean("verbose")
            self.report_period = int(config["report_period"])
            self.save_period = int(config["save_period"])

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
