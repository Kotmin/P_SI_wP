import numpy as np
import pandas as pd

import os


from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import random


import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,Activation, Dropout,GaussianNoise
from tensorflow.keras.layers import Input,UpSampling2D, BatchNormalization


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential



def show_project_dir_info():
    for dirpath, dirnames, filenames in os.walk("./"):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in {dirpath}")


    num_of_coffe_bean_classes = len(os.listdir("./train"))
    print(num_of_coffe_bean_classes)

# Dane jakby się zgadzały 1600 obrazow w formacie png

def show_radom_image(target_dir, target_class):
    target_folder = target_dir + target_class
    # random path
    random_image = random.sample(os.listdir(target_folder),1)

    img = mpimg.imread(target_folder + "/" + random_image[0])
    plt.imshow(img)
    plt.title(target_class)
    plt.axis("off")
    # plt.show()

    print(f"Image shape: {img.shape}")

    return img



train_dir = "./train/"
test_dir = "./test/"

plt.figure(figsize=(15,7))
plt.subplot(1,4,1)
dark_bean_img = show_radom_image("train/",target_class="Dark")
plt.subplot(1,4,2)
green_bean_img = show_radom_image("train/",target_class="Green")
plt.subplot(1,4,3)
glight_bean_img = show_radom_image("train/",target_class="Light")
plt.subplot(1,4,4)
medium_bean_img = show_radom_image("train/",target_class="Medium")

plt.show()

