# Import data from local file.

import shutil
import matplotlib.pyplot as plt
import os
import glob
import random
from matplotlib.image import imread

# Images path

path_norbert = r"D:\Fire_project\Fire_project\fire_dataset"
path_alicja = r""
path_dagmara = r""

# Project path

path_project_norbert = r"C:\Users\NorbertK\PycharmProjects\Fire_detection_project"
path_project_alicja = r""
path_project_dagmara = r""

# Pick your name

path_check = "Norbert"

# Loop to select current path

if path_check == "Norbert":
    path = path_norbert
    path_project = path_project_norbert
elif path_check == "Alicja":
    path = path_alicja
    path_project = path_project_alicja
else:
    path = path_dagmara
    path_project = path_project_dagmara

# Organize data into train, valid, test dirs

os.chdir(path)
if os.path.isdir('train/fire') is False:
    os.makedirs('train/fire')
    os.makedirs('train/non_fire')
    os.makedirs('valid/fire')
    os.makedirs('valid/non_fire')
    os.makedirs('test/fire')
    os.makedirs('test/non_fire')

    for c in random.sample(glob.glob('fire*'), 100):
        shutil.move(c, 'train/fire')
    for c in random.sample(glob.glob('non_fire*'), 100):
        shutil.move(c, 'train/non_fire')
    for c in random.sample(glob.glob('fire*'), 50):
        shutil.move(c, 'valid/fire')
    for c in random.sample(glob.glob('non_fire*'), 50):
        shutil.move(c, 'valid/non_fire')
    for c in random.sample(glob.glob('fire*'), 20):
        shutil.move(c, 'test/fire')
    for c in random.sample(glob.glob('non_fire*'), 20):
        shutil.move(c, 'test/non_fire')

# Plotting random 16 original images (fire)

plt.figure(figsize=(12,12))
path_train = path + r"\train\fire"
for i in range (1, 17):
    plt.subplot(4, 4, i)
    plt.tight_layout()
    rand_img = imread(path_train + '/' + random.choice(sorted(os.listdir(path_train))), format='jpg')
    plt.imshow(rand_img)
    plt.xlabel(rand_img.shape[1], fontsize = 10)
    plt.ylabel(rand_img.shape[0], fontsize = 10)
    plt.savefig(path_project + r"\reports\figures\sample_fire.png")

# Plotting random 16 original images (non-fire)

plt.figure(figsize=(12,12))
path_train = path + r"\train\non_fire"
for i in range (1, 17):
    plt.subplot(4, 4, i)
    plt.tight_layout()
    rand_img = imread(path_train + '/' + random.choice(sorted(os.listdir(path_train))), format='jpg')
    plt.imshow(rand_img)
    plt.xlabel(rand_img.shape[1], fontsize = 10)
    plt.ylabel(rand_img.shape[0], fontsize = 10)
    plt.savefig(path_project + r"\reports\figures\sample_non_fire.png")

