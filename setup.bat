conda create --name GANEnv --file requirements.txt

call activate GANEnv

conda install -c peterjc123 pytorch cuda90
pip install torchvision