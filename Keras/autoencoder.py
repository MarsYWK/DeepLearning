
# coding: utf-8

# In[15]:


from keras.layers import Input, Dense
from keras.models import Model

from keras.datasets import mnist
import numpy as np


# In[ ]:


#这是我们编码表示的大小
encoding_dim = 32

#输入占位符
input_img = Input(shape=(784,))
#输入的编码表示
encoded = Dense(encoding_dim, activation='relu')(input_img)
#“解码”是输入的有损重建
decoded = Dense(784, activation='sigmoid')(encoded)

#这个模型将输入映射到其重建
autoencoder = Model(inputs=input_img, outputs=decoded)


encoder = Model(inputs=input_img, outputs=encoded)
#为编码（32维）输入创建占位符
encoded_input = Input(shape=(encoding_dim,))
#检索自动编码器模型的最后一层
decoder_layer = autoencoder.layers[-1]
#创建解码器模型
decoder = Model(inputs=encoded_input, outputs=decoder_layer(encoded_input))


# In[17]:


autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


# In[18]:


#准备MNIST数据，将其归一化和向量化
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)


# In[19]:


#训练
autoencoder.fit(x_train, x_train,
               epochs=50,
               batch_size=256,
               shuffle=True,
               validation_data=(x_test, x_test))


# In[20]:


import matplotlib.pyplot as plt


# In[29]:


encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

n = 10
plt.figure(figsize=(20,4))
for i in range(1,11):
    #原始图片
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    #重构图片
    ax = plt.subplot(2, n, i+n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# In[30]:


from keras import regularizers


# In[33]:


#稀疏自编码器：加上稀疏性约束
encoding_dim = 32

input_img = Input(shape=(784,))
encoded = Dense(encoding_dim, activation='relu',
               activity_regularizer=regularizers.activity_l1(10e-5))(input_img)
decoded = Dense(784, activation='sigmoid')(encoded)
autoencoder = Model(inputs=input_img,outputs=decoded)


# In[35]:


#深度自编码器：把自编码器叠起来
input_img = Input(shape=(784,))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)

autoencoder = Model(inputs=input_img, outputs=decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train,
               epochs=100,
               batch_size=256,
               shuffle=True,
               validation_data=(x_test, x_test))


# In[37]:


#卷积自编码器
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model

input_img = Input(shape=(1, 28, 28))

x = Convolution2D(16, 3, 3, activation='relu', padding='same')(input_img)
x = MaxPooling2D((2,2), padding='same')(x)
x = Convolution2D(8, 3, 3, activation='relu', padding='same')(x)
x = MaxPooling2D((2,2), padding='same')(x)
x = Convolution2D(8, 3, 3, activation='relu', padding='same')(x)
encoded = MaxPooling2D((2,2), activation='same')

x = Convolution2D(8, 3, 3, activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(8, 3, 3, activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(16, 3, 3, activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Convolution2D(1, 3, 3, activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


# In[38]:


#序列与序列的自动编码器
from keras.layers import Input, LSTM, RepeatVector
from keras.models import Model

inputs = Input(shape=(timesteps, input_dim))
encoded = LSTM(latent_dim)(inputs)

decoded = RepeatVector(timesteps)(encoded)
decoded = LSTM(input, return_sequences=True)(decoded)

sequence_autoencoder = Model(inputs, decoded)
encoder = Model(inputs, encoded)


# In[ ]:


#变分自编码器（Variational autoencoder，VAE）：编码数据的分布

#首先：建立编码网络，将输入影射为隐分布的参数
x = Input(batch_shape=(batch_size, original_dim))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_sigma = Dense(latent_dim)(h)


#然后从这些参数确定的分布中采样，这个样本相当于之前的隐层值
def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim),
                             mean=0., std=epsilon_std)
    return z_mean + K.exp(z_log_sigma) * epsilon

#注意，TensorFlow后端不需要“output_shape”
#所以你可以写'Lambda（sampling）（[z_mean，z_log_sigma]）`
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean,z_log_sigma])

#最后，将采样得到的点映射回去重构原输入：
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

