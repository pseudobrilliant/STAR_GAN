#!/bin/bash

sudo apt-get install git python3-pip python3-dev build-essential swig python-wheel libcurl3-dev libfreetype6-dev libpng12-dev

sudo apt-get update

sudo pip3 install virtualenv

virtualenv --system-site-packages -p python3 ./venv

source ./venv/bin/activate

sudo pip3 install -r requirements.txt

sudo pip3 install -r torch

sudo pip3 install -r torchvision