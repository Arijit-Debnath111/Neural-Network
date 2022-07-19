#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import LSTM
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


#Timeseries


# In[3]:


np.array(range(0,100))/100 


# In[4]:


Data = [[[(i+j)/100] for i in range(5)] for j in range(100)]
Data[:5]


# In[5]:


Target = [(i+5)/100 for i in range(100)]
Target[:5]


# In[6]:


data = np.array(Data,dtype=float)
target = np.array(Target,dtype=float)


# In[7]:


data.shape, target.shape


# In[8]:


data


# In[9]:


target


# In[10]:


#Dividing data into train & test


# In[11]:


x_train, x_test, y_train, y_test = train_test_split(data,target,test_size=0.2,random_state=4)


# In[12]:


x_train


# In[13]:


#RNN


# In[14]:


model = Sequential() 
model.add(SimpleRNN((5),batch_input_shape=(None,5,1),return_sequences=False, activation='relu'))
model.add(Dense(1))


# In[15]:


model.compile(loss='mean_squared_error',optimizer='adam',metrics=['mean_squared_error'])


# In[16]:


model.summary()


# In[17]:


history = model.fit(x_train,y_train,epochs=100,validation_data=(x_test,y_test))


# In[18]:


results = model.predict(x_test)


# In[19]:


plt.scatter(range(20),results,c='r')
plt.scatter(range(20),y_test,c='g')
plt.legend(['Actual','Predicted'])
plt.show()


# In[20]:


plt.plot(history.history['loss'], label= 'Train Loss')
plt.plot(history.history['val_loss'], label= 'Val Loss')
plt.legend()
plt.show()


# In[ ]:




