#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install keras.utils')


# In[4]:


from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt 
from tensorflow.keras.utils import to_categorical


# In[5]:


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


# In[6]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[7]:


y_train_ohe = to_categorical(y_train)
y_test_ohe = to_categorical(y_test)


# In[8]:


x_train = x_train.reshape(60000,28,28,1).astype(float)
x_test = x_test.reshape(10000,28,28,1).astype(float)


# In[9]:


# Exploring Dataset


# In[10]:


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# In[14]:


plt.figure(figsize=(10,12))
for i in range(10):
    plt.subplot(5,5,i+1)
    plt.imshow(x_train[i],cmap='gray')
    plt.title(class_names[y_train[i]])
plt.show()


# In[15]:


# Model Building


# In[17]:


from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPool2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam


# In[18]:


model = Sequential()
model.add(Conv2D(filters=10,kernel_size=(3,3),strides=(1, 1),padding='valid'))
model.add(Activation('relu'))

model.add(Conv2D(filters=10,kernel_size=(3,3),strides=(1, 1),padding='valid'))
model.add(Activation('relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(10, activation='softmax'))


# In[19]:


model.compile(loss='categorical_crossentropy',optimizer = Adam(learning_rate=0.01),metrics = ['accuracy'])


# In[20]:


model.fit(x=x_train,y=y_train_ohe,batch_size=1000,validation_data=(x_test, y_test_ohe),epochs = 15)


# In[21]:


train_loss = model.history.history['loss']
val_loss = model.history.history['val_loss']
train_accuracy = model.history.history['accuracy']
validation_accuracy = model.history.history['val_accuracy']


# In[22]:


plt.plot(train_accuracy, label='Train')
plt.plot(validation_accuracy,label='Validation')
# plt.ylim(0,1.1)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curves")
plt.legend()
plt.grid()
plt.show()


# In[23]:


plt.plot(train_loss, label='Train')
plt.plot(val_loss,label='Validation')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curves")
plt.legend()
plt.grid()
plt.show()


# In[ ]:




