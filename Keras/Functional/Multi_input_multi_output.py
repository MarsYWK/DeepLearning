
# coding: utf-8

# In[9]:


from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
import keras


# In[10]:


#标题输入：意味着接收100个整数的序列，介于1到10000之间。
#请注意，我们可以通过传递“name”参数命名任何图层。
main_input = Input(shape=(100,), dtype='int32',name='main_input')

#该嵌入层将输入序列编码成密集的512维向量序列。
x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)

#LSTM将向量序列变换为单个向量，其中包含有关整个序列的信息
lstm_out = LSTM(32)(x)


# In[11]:


#插入一个额外的损失，使得即使在主损失很高的情况下，
#LSTM和Embedding层也可以平滑的训练。
auxiliary_output = Dense(1, activation='sigmoid',name='aux_output')(lstm_out)


# In[12]:


#讲LSTM与额外的输入数据串联起来组成输入，送入模型
auxiliary_input = Input(shape=(5,), name='aux_input')
x = keras.layers.concatenate([lstm_out, auxiliary_input])

#全链接层
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

#最后我们添加主逻辑回归层
main_output = Dense(1, activation='sigmoid', name='main_output')(x)


# In[13]:


#定义整个2输入，2输出的模型
model = Model(inputs = [main_input, auxiliary_input], 
             outputs = [main_output, auxiliary_output])


# In[14]:


model.compile(optimizer='rmsprop',loss='binary_crossentropy',
             loss_weights=[1., 0.2])


# In[15]:


model.fit([Headline_data, additional_data],
         [labels,labels],
         epochs=50, batch_size=32)


# In[ ]:


#我们输入和输出是被命名过的（在定义时传递了“name”参数），
#我们也可以用下面的方式编译和训练模型：
model.compile(optimizer='rmsprop',
             loss={'main_output':'binary_crossentropy'},
             'aux_output':'binary_crossentropy')

model.fit({'main_input': headline_data,'aux_input':additional_data},
         {'main_output':labels, 'aux_output':labels},
         epochs=50, batch_size=32)

