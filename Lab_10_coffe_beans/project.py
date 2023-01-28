import os

import numpy as np

from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import random


import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,Activation, Dropout,GaussianNoise
from tensorflow.keras.layers import Input,UpSampling2D, BatchNormalization


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential, preprocessing



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

def show_examples()->None:

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


train_dir = "./train/"
test_dir = "./test/"


train_datagen = ImageDataGenerator( rescale = 1./255)
test_datagen = ImageDataGenerator( rescale = 1./255)


# rozwaz 256   
train_data = train_datagen.flow_from_directory( directory= train_dir,
                                                batch_size=32, 
                                                target_size=(224,224),
                                                class_mode="categorical",
                                              )
                                              
                                              
test_data = test_datagen.flow_from_directory( directory= test_dir,
                                                batch_size=32, 
                                                target_size=(224,224),
                                                class_mode="categorical",
                                            )

# print(train_data)
# print(test_data)

print(train_data[1][0]) # tabella z podziałem kazdy pixel opisany z notacja kolorystyczna kształ tabel(32, 224, 224, 3)
#przyjmuje wartosci (0-1)
print(train_data[1][1]) # kod pierscieniowy / zapewne cechy
# sumarycznie jest to tuple o dwoch elementach


base_model = tf.keras.applications.ResNet50V2(include_top = False)

base_model.trainable = False
inputs = tf.keras.layers.Input(shape = (224,224,3), name = "input-layer")
x=tf.keras.layers.experimental.preprocessing.Rescaling(1/255.)(inputs)

x = base_model(inputs)
print(f"Shape after passing inputs thr base model: {x.shape}")

#6
x = tf.keras.layers.GlobalAveragePooling2D(name= "global_average_pooling_layer")(x)
print(f"Shape after GlobalAvrPool@d: {x.shape}")

outputs = tf.keras.layers.Dense(4, activation = "softmax", name = "output-layer")(x)

model_0 = tf.keras.Model(inputs, outputs)

model_0.compile(loss = "categorical_crossentropy",
                optimizer = Adam(learning_rate=0.001),
                metrics = ["accuracy"]
               )

history = model_0.fit(train_data,
                      epochs=5,
                      steps_per_epoch = len(train_data),
                      validation_data = test_data,
                      validation_steps = int(0.25*len(test_data)),
                     )               


model_0.summary()                     


## Czas spróbować wygenerować jakieś dane

list_of_classes = ["Dark","Green","Light","Medium"]

def get_prediction(object):
    pred =  model_0.predict(tf.expand_dims(object,axis=0))

    if len(pred[0]) > 1:
        pred_class = list_of_classes[pred.argmax()]
    else:
        pred_class = list_of_classes[int(tf.round(pred)[0][0])]

    # plt.imshow(object)
    plt.title(f"Prediction: {pred_class}")
    print(f"Prediction: {pred_class}")
    return object


def show_test_of_noisiness_reduction(model_name):
    

    sampples = []

    for _ in range(5):
        tar=random.choice(list_of_classes)
        print(tar)
        sampples.append(show_radom_image("test/", target_class=tar ))

    # test_photos = sampples.copy()
    # mask = np.random.randn(*test_photos.shape)
    # white = mask > 1
    # black = mask < -1
    # test_photos[white] = 255
    # test_photos[black] = 0
    # test_photos /= 255

    plt.figure(figsize=(15,7))

    plt.subplot(2,5,1)
    t1 = sampples[0].copy()
    plt.subplot(2,5,2)
    t2 = sampples[1].copy()
    plt.subplot(2,5,3)
    t3 = sampples[2].copy()
    plt.subplot(2,5,4)
    t4 = sampples[3].copy()
    plt.subplot(2,5,5)
    t5 = sampples[4].copy()


    # Prediction

    plt.subplot(2,5,6)
    t = get_prediction(sampples[0])
    plt.subplot(2,5,7)
    t = get_prediction(sampples[1])
    plt.subplot(2,5,8)
    t = get_prediction(sampples[2])
    plt.subplot(2,5,9)
    t = get_prediction(sampples[3])
    plt.subplot(2,5,10)
    t = get_prediction(sampples[4])

    plt.show()

show_test_of_noisiness_reduction(model_0)