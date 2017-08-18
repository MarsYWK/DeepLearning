
# coding: utf-8

# In[9]:


from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np


# In[10]:


data_dim = 16
timesteps = 8
num_classes = 10


# In[11]:


model = Sequential()
#返回尺寸为32的向量序列
model.add(LSTM(32, return_sequences=True,
              input_shape=(timesteps, data_dim)))
#返回尺寸为32的向量序列
model.add(LSTM(32, return_sequences=True))
#返回一个维度为32的单个向量
model.add(LSTM(32))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

#训练集
x_train = np.random.random((1000,timesteps,data_dim))
y_train = np.random.random((1000,num_classes))

#验证集
x_val = np.random.random((100,timesteps,data_dim))
y_val = np.random.random((100,num_classes))

model.fit(x_train,y_train,
         batch_size=64,epochs=5,
         validation_data=(x_val,y_val))




# In[ ]:




