import os
import configparser
import src.acgan_discriminator
import src.acgan_generator
import torchvision.datasets as data
import torchvision.transforms as transforms
import torch.utils.data.Dataloader as datal

config = configparser.ConfigParser()
path = os.path.abspath("./")
config.read(os.path.join(path, "config.ini"))

image_size = config["GLOBAL"]["Image_Size"]

transform = transforms.Compose([
    transforms.CenterCrop(image_size * 2),
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = data.ImageFolder(root='./data', download=True)
loader = datal(dataset, batch_size=64, shuffle=True, num_workers=4, transforms=transform)









