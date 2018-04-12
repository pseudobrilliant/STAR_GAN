import os
import torchvision.datasets as data
import configparser
from src.acgan import ACGAN


import os

config = configparser.ConfigParser()
path = os.path.abspath("./")
config.read(os.path.join(path, "config.ini"))

acgan_config = config["ACGAN"]
acgan = ACGAN(acgan_config)

acgan.GenerateSampleImages("./saves/final_samples")
