#!/usr/bin/env python
# coding: utf-8

# In[1]:


import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os
from PIL import Image
import pathlib
import csv


# In[2]:


# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# In[3]:


#Keras
import keras


# In[4]:


import warnings
warnings.filterwarnings('ignore')


# In[5]:


d=pd.read_csv('prj_data.csv')


# In[6]:


d


# In[7]:


d=d.drop(['filename'],axis=1)


# In[8]:


types_list = d.iloc[:, -1]
onehot_le = LabelEncoder()
p = onehot_le.fit_transform(types_list)


# In[9]:


di_std = StandardScaler()


# In[10]:


r = di_std.fit_transform(np.array(d.iloc[:, :-1], dtype = float))


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(r, p, test_size=0.25)


# In[55]:


X_train[10]


# In[13]:


from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)))

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(10, activation='softmax'))


# In[14]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[15]:


history = model.fit(X_train,
                    y_train,
                    epochs=20,
                    batch_size=128)


# In[16]:


test_loss, test_acc = model.evaluate(X_test,y_test)


# In[17]:


print('test_acc: ',test_acc)


# In[18]:


x_val = X_train[:200]
partial_x_train = X_train[200:]

y_val = y_train[:200]
partial_y_train = y_train[200:]


# In[49]:


model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(partial_x_train,
          partial_y_train,
          epochs=30,
          batch_size=512,
          validation_data=(x_val, y_val))
results = model.evaluate(X_test, y_test)


# In[50]:


results


# In[22]:


predictions = model.predict(X_test)


# In[23]:


predictions[0].shape


# In[24]:


np.sum(predictions[0])


# In[25]:


np.argmax(predictions[0])


# In[ ]:




