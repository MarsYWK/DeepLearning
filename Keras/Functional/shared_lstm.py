
# coding: utf-8

# In[1]:


import keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model

tweet_a = Input(shape=(140,256))
tweet_b = Input(shape=(140,256))


# 若要对不同的输入共享同一层，就初始化该层一次，然后多次调用它

# In[2]:


#此层可以作为输入矩阵，并返回大小为64的向量
shared_lstm = LSTM(64)

#当我们重复使用相同的层实例多次时，
#层的权重也被重用（它实际上是*相同的*层）
encoded_a = shared_lstm(tweet_a)
encoded_b = shared_lstm(tweet_b)

#我们可以连接两个向量：
merged_vector = keras.layers.concatenate([encoded_a, encoded_b], axis=-1)

#在顶部添加逻辑回归
predictions = Dense(1, activation='sigmoid')(merged_vector)

#我们定义一个可传达的模型，将tweet输入与预测相联系
model = Model(inputs=[tweet_a, tweet_b], outputs=predictions)

model.compile(optimizer='rmsprop',
             loss='binary_crossentropy',
             metrics=['accuracy'])

model.fit([data_a,data_b], labes, epochs=10)


# In[ ]:




