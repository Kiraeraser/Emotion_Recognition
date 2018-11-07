
# coding: utf-8

# In[1]:


import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Activation, Dropout, Flatten

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt

import sys
from scipy.misc import imresize


# In[2]:


import numpy as np

from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
from keras.models import Sequential
get_ipython().run_line_magic('matplotlib', 'inline')
np.random.seed(7)
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)


# In[11]:


from keras.layers.convolutional import ZeroPadding2D


# In[3]:


config = tf.ConfigProto( device_count = {'GPU': 0 , 'CPU': 56} ) #max: 1 gpu, 56 cpu
sess = tf.Session(config=config) 
keras.backend.set_session(sess)


# In[4]:


num_classes = 7 #angry, disgust, fear, happy, sad, surprise, neutral
batch_size = 256
epochs = 12


# In[5]:


label={0:"Angry", 1:"Disgust", 2:"Fear", 3:"Happy", 4:"Sad", 5:"Surprise", 6:"Neutral"}


# In[6]:


with open("fer2013.csv") as f:
    content = f.readlines()

lines = np.array(content)

num_of_instances = lines.size
print("number of instances: ",num_of_instances)
print("instance length: ",len(lines[1].split(",")[1].split(" ")))


# In[7]:


x_train, y_train, x_test, y_test = [], [], [], []


# In[8]:


for i in range(1,num_of_instances):
    try:
        emotion, img, usage = lines[i].split(",")
          
        val = img.split(" ")
            
        pixels = np.array(val, 'float32')
        
        emotion = keras.utils.to_categorical(emotion, num_classes)
    
        if 'Training' in usage:
            y_train.append(emotion)
            x_train.append(pixels)
        elif 'PublicTest' in usage:
            y_test.append(emotion)
            x_test.append(pixels)
    except:
        print("",end="")


# In[9]:


x_train = np.array(x_train, 'float32')
y_train = np.array(y_train, 'float32')
x_test = np.array(x_test, 'float32')
y_test = np.array(y_test, 'float32')

x_train /= 255 #normalize inputs between [0, 1]
x_test /= 255

x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
x_train = x_train.astype('float32')
x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)
x_test = x_test.astype('float32')

print(x_train.shape, 'train samples')
print(x_test.shape, 'test samples')


# In[14]:


def ERModel():
    model = Sequential()
    
    #1st convolution layer
    model.add(Conv2D(64, (5, 5), border_mode='valid', activation='relu', input_shape=(48,48,1)))
    model.add(ZeroPadding2D(padding=(2, 2), dim_ordering='tf'))
    model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))

    #2nd convolution layer
    model.add(ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

    #3rd convolution layer
    model.add(ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
    model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

    model.add(Flatten())

    #fully connected neural networks
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))

    #output Softmax
    model.add(Dense(num_classes, activation='softmax'))
   


    return model


# In[15]:


model=ERModel()


# In[ ]:


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[15]:


model.save('ER_model_training12ep.h5')

