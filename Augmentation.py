#jonathan mairena
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 16:19:11 2020

@author: jonathanmairena
"""


import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl



transformations = transforms.Compose([
    transforms.Resize((255,255)),
    transforms.ColorJitter(saturation=.05),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20,resample= Image.BILINEAR),
    #transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#train_set = datasets.ImageFolder(r"/Users/jonathanmairena/Documents/Fall2020/SeniorDesign/Training-validation", transform = transformations)

test_set = datasets.ImageFolder(r"/Users/jonathanmairena/Documents/Fall2020/SeniorDesign/New-Test", transform = transformations)

mpl.rcParams['axes.grid'] = False
mpl.rcParams['image.interpolation'] = 'nearest'
mpl.rcParams['figure.figsize'] = 15, 25

def show_dataset(dataset, n=6):
  img = np.vstack((np.hstack((np.asarray(dataset[i][0]) for _ in range(n)))
                   for i in range(len(dataset))))
  plt.imshow(img)
  plt.axis('off')

show_dataset(test_set)