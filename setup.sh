#!/bin/bash

sudo apt-get install git python3-pip python3-dev build-essential swig python-wheel libcurl3-dev libfreetype6-dev libpng12-dev

sudo apt-get update

pip3 install virtualenv

virtualenv --system-site-packages -p python3 ./venv

source venv/bin/activate

pip3 install -r requirements.txt

pip3 install -r torch

pip3 install -r torchvision