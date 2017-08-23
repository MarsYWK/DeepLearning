
# coding: utf-8

# In[8]:


from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense


# In[9]:


#模型权重文件路径
weigths_path = '../keras/examples/vgg16_weights.h5'
top_model_weights_path = 'fc_model.h5'
#图片尺寸
img_width, img_height = 150, 150

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 50
batch_size = 16


# In[13]:


#建立VGG16网络
model = applications.VGG16(weights='imagenet', include_top = False)
print('Model loaded.')

#构建一个分类器模型，以摆在卷积模型的顶端
top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))


# In[12]:


#注意，有必要从一个全训练的分类器开始，
#包括顶级分类器，以便成功地进行微调
top_model.load_weights(top_model_weights_path)

#将模型添加到卷积基础之上
model.add(top_model)

#将前25个层（最后一个转换块）设置为不可训练（权重不会更新）
for layer in model.layers[:25]:
    layer.trainable = False

#用SGD /动量优化器编译模型,很慢的学习速度
model.compile(loss='binary_crossentrop',
             optimizer=optimizers.SGD(lr=1e-4,momentum=0.9),
             metrics=['accuracy'])

#准备数据扩充配置
train_datagen = ImageDataGenerator(
    rescale=1. / 255, #图像值设为0~1之间的书
    shear_range=0.2, #剪切
    zoom_range=0.2, #放大
    horizontal_flip=True) #水平旋转

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_model='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_model='binary')

#微调模型
model.fit_generator(train_generator,
                   samples_per_epoch=nb_train_sampes,
                   epochs=epochs,
                   validation_data=validation_generator,
                   nb_val_samples=nb_validation_samples)


# In[ ]:




