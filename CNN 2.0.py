#!/usr/bin/env python
# coding: utf-8

# In[2]:


from keras.datasets  import cifar10
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical


# In[3]:


(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# In[4]:


x_train.shape


# In[5]:


# Visualise


# In[6]:


y_train[2]


# In[7]:


class_names = ['airplane', 'automobile', 'bird', 'cat','deer','dog','frog','horse','ship','truck']


# In[8]:


plt.figure(figsize=(12,15))
for i in range(10):
    plt.subplot(5, 5, i+1)
    plt.imshow(x_train[i])
    plt.title(class_names[y_train[i][0]])
plt.show()


# In[9]:


# Preparing Datasets


# In[10]:


y_train_ohe = to_categorical(y_train)
y_test_ohe = to_categorical(y_test)
x_train = x_train.reshape(-1,32,32,3).astype('float')
x_test = x_test.reshape(-1,32,32,3).astype('float')


# In[11]:


y_train_ohe


# In[12]:


x_train[0].shape


# In[13]:


#Model Building


# In[15]:


from keras.models import Sequential
from keras.layers import Conv2D, Dense, Activation, MaxPool2D, Flatten, Input, Dropout
from tensorflow.keras.optimizers import RMSprop, Adam


# In[16]:


model = Sequential()
model.add(Conv2D(filters=10, kernel_size = (3,3), strides = (1,1)))
model.add(Activation('relu')) 

model.add(Conv2D(filters=10, kernel_size = (3,3), strides = (1,1)))
model.add(Activation('relu')) 

model.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1))) # 14x14
model.add(Flatten())
# model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax')) 


# In[17]:


model.compile(loss='categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])


# In[18]:


model.fit(x=x_train,y=y_train_ohe,batch_size=1000,validation_data=(x_test, y_test_ohe),epochs = 10)


# In[19]:


train_loss = model.history.history['loss']
val_loss = model.history.history['val_loss']
train_accuracy = model.history.history['accuracy']
validation_accuracy = model.history.history['val_accuracy']


# In[20]:


plt.plot(train_loss)
plt.plot(val_loss)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("LOSS CURVE")
plt.legend(['Train', 'Validation'])
plt.grid()
plt.show()


# In[21]:


plt.plot(train_accuracy)
plt.plot(validation_accuracy)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("ACCURACY CURVE")
plt.legend(['Train', 'Validation'])
plt.grid()
plt.show()


# In[ ]:




