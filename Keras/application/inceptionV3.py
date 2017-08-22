
# coding: utf-8

# In[2]:


from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K


# In[6]:


#创建基本的预训练模型
base_model = InceptionV3(weights='imagenet', include_top=False)

#添加一个全局空间平均池化层
x = base_model.output
x = GlobalAveragePooling2D()(x)
#增加全连接层
x = Dense(1024, activation='relu')(x)
#一个逻辑层 - 假设我们有200个类
predictions = Dense(200, activation='softmax')(x)

#我们要训练的模型
model = Model(inputs=base_model.input, outputs=predictions)

#第一：只训练顶层（随机初始化）
#即冻结所有卷积InceptionV3图层
for layer in base_model.layers:
    layer.trainable = False

#编译模型（应该完成*之后*将图层设置为不可训练）
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

#在这个新数据上训练这个模型
model.fit_generator(...)

#在这一点上，顶层是受过良好训练的，
#我们可以从起始V3开始微调卷积层。 
#我们将冻结底层N层，并训练其余顶层。

#让我们可视化图层名称和图层索引，
#看看我们应该冻结多少图层：
for layer in model.layers[:172]:
    layer.trainable = False
for layer in model.layers[172:]:
    layer.trainable = True
    
#我们需要重新编译这些修改的模型才能生效，
#我们以低学习率使用SGD
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9,
                           loss='categorical_crossentropy'))

#我们再次训练我们的模型（这次微调顶级密集层的顶部2个起始块
model.fit_generator(...)



# In[ ]:




