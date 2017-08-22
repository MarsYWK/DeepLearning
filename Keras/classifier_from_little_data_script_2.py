
# coding: utf-8

# In[3]:


from keras.applications.vgg16 import VGG16
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense


# In[4]:


#图片的尺寸
img_width, img_height = 150, 150

top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 50
batch_size = 16


# In[7]:


def save_bottleneck_features():
    
    datagen = ImageDataGenerator(rescale=1. / 255)
    
    #建立VGG16网络
    model = VGG16(weights='imagenet', include_top=False)


    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size = batch_size,
        class_mode = None,#这意味着我们的生成器只会产生批量的数据，没有标签
        shuffle=False)#我们的数据将是有序的，所以所有前1000个图像将是猫，然后1000个dog，
    #predict_generator方法返回一个模型的输出，给出一个产生批次数字数据的生成器

    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    #将输出保存为Numpy数组
    np.save(open('bottleneck_features_train.npy', 'w'),
            bottleneck_features_train)
    
    generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width,img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save(open('bottleneck_features_validation.npy','w'),
            bottleneck_features_validation)


# In[ ]:


def train_top_model():
    
    #记录完毕后我们可以将数据载入，用于训练我们的全连接网络：
    train_data = np.load(open('bottleneck_features_train.npy'))
    #特征按顺序保存，因此重新创建标签很容易
    train_labels = np.array(
        [0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))

    validation_data = np.load(open('bottleneck_features_validation.npy'))
    validation_labels = np.array(
        [0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2))

    model = Sequential()
    model.add(Flatten(input_shape=train_data,shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer = 'rmsprop',
                 loss = 'binary_crossentropy',
                 metrics = ['accuracy'])

    model.fit(train_data,train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)
    
save_bottleneck_features()
train_top_model()

