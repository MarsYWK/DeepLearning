
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np


# In[2]:


data_dim = 16
timesteps = 8
num_classes = 10
batch_size = 32


# In[3]:


#预期的输入批次形状：（batch_size，timesteps，data_dim）
#注意，由于网络是有状态的，我们必须提供完整的batch_input_shape。 
#批次k中的索引i的样本是批次k-1中样本i的后续。
model = Sequential()
model.add(LSTM(32, return_sequences=True, stateful=True,
              batch_input_shape=(batch_size, timesteps, data_dim)))
model.add(LSTM(32, return_sequences=True, stateful=True))
model.add(LSTM(32, stateful=True))
model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',
             optimizer='rmsprop',
             metrics=['accuracy'])

#训练集
x_train = np.random.random((batch_size * 10, timesteps, data_dim))
y_train = np.random.random((batch_size * 10, num_classes))

#测试集
x_val = np.random.random((batch_size * 3, timesteps, data_dim))
y_val = np.random.random((batch_size * 3, num_classes))

model.fit(x_train, y_train,
         batch_size=batch_size, epochs=5, shuffle=False,
         validation_data=(x_val, y_val))


# In[ ]:




