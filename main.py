import os
import configparser
from src.acgan import ACGAN
import torchvision.transforms as transforms

import os

config = configparser.ConfigParser()
path = os.path.abspath("./")
config.read(os.path.join(path, "config.ini"))

image_size = int(config["GLOBAL"]["image_size"])
transform = transforms.Compose([
    transforms.CenterCrop(image_size * 2),
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

acgan_config = config["ACGAN"]
ACGAN(acgan_config, transform)






