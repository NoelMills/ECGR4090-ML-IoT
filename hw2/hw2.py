#!/usr/bin/env python
# coding: utf-8

#In[1]
epochs = 50
activation_func = "relu"
# In[2]:
# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10
from tensorflow.keras import Input, layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
print(tf.__version__)

#In[3]
(train_images, train_labels),(test_images,test_labels) = tf.keras.datasets.cifar10.load_data()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship','truck']
input_shape = train_labels.squeeze()
print(input_shape)

#In[4]
idx = 10
plt.figure()
plt.imshow(train_images[idx])
plt.colorbar()
plt.title("Label = {:}".format(class_names[train_labels[idx][0]]))

#In[5]
#Part 5

model = Sequential()

model.add(Conv2D(32, (3,3), strides = (2,2), padding = 'same', activation = 'relu', input_shape = (32, 32, 3)))
model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
model.add(Conv2D(64, (3,3), strides = (2,2), padding = 'same', activation = 'relu'))
model.add(Flatten())
model.add(Dense(1024, activation = 'relu'))
model.add(Dense(10))

model.summary()
# 

#In[6]
#Part 6
model.compile(optimizer='adam', 
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
train_hist = model.fit(train_images, train_labels, epochs=50, validation_data=(test_images, test_labels))
test_lost, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy: ', test_acc)

#In[7]
from tensorflow.keras.preprocessing import image
import PIL
import PIL.Image
timage = image.load_img('kat.jpg', target_size=(32,32))
timage = image.img_to_array(timage)
timage = np.array([timage/255])
predict = model.predict(timage,batch_size=1)
print(predict)
predict_class = model.predict_classes(timage,batch_size = 1)
print(predict_class)
# %%

#In[8]
model2 = Sequential()

model2.add(tf.keras.layers.DepthwiseConv2D(kernel_size= (3,3), strides= (2,2), padding= 'same', input_shape = (32, 32, 3)))
model2.add(Conv2D(32, (3,3), strides = (2,2), padding = 'same', activation = 'relu'))
model2.add(tf.keras.layers.DepthwiseConv2D(kernel_size= (3,3), strides= (2,2), padding= 'same'))
model2.add(Conv2D(64, (1,1), strides = (1,1), padding = 'same', activation = 'relu'))


model2.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
model2.add(Flatten())
model2.add(Dense(1024, activation = 'relu'))
model2.add(Dense(10))

model2.summary()