from utils.utils import mkdirs
import os

path = 'datasets/VGGface2_HQ/'

train_path = os.path.join(path, 'train')
test_path = os.path.join(path, 'test')

for d in os.listdir(os.path.join(train_path, 'images')):
    path = os.path.join(train_path, 'latent-ID')
    path = os.path.join(path, d)
    mkdirs(path)

for d in os.listdir(os.path.join(test_path, 'images')):
    path = os.path.join(test_path, 'latent-ID')
    path = os.path.join(path, d)
    mkdirs(path)