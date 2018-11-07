
# coding: utf-8

# In[27]:


import tensorflow as tf

import keras
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, Dropout
from keras.models import Sequential
from keras.models import Model, load_model
from keras.utils import layer_utils
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
np.random.seed(7)
import sys
from scipy.misc import imresize


# In[2]:


config = tf.ConfigProto( device_count = {'GPU': 0 , 'CPU': 56} ) #max: 1 gpu, 56 cpu
sess = tf.Session(config=config) 
keras.backend.set_session(sess)


# In[3]:


num_classes = 7 #angry, disgust, fear, happy, sad, surprise, neutral
batch_size = 256
epochs = 12


# In[4]:


label={0:"Angry", 1:"Disgust", 2:"Fear", 3:"Happy", 4:"Sad", 5:"Surprise", 6:"Neutral"}


# In[10]:


model = load_model('ER_model_training12ep.h5')




# In[12]:


def predict_emotion(img):
    plt.figure()
    plt.imshow(img, cmap="gray")
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    x /= 255
    plt.imshow(img)
    custom = model.predict(x)

    i=0
    l=[]
    print("Analysis")
    print("-"*20)
    for j in np.nditer(custom):
        print (label[i],"->",j*100,"%")
        l.append(j)
        i=i+1
    print("-"*20)
    print("\nThis Image is",label[l.index(max(l))])


# In[ ]:


img = image.load_img(input("Enter the name of the image: "), color_mode = "grayscale", target_size=(48, 48))
x=imresize(img,(48,48),'nearest')
predict_emotion(x)
plt.imshow(x,cmap='gray')

