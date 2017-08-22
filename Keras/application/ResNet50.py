
# coding: utf-8

# In[2]:


from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np


# In[5]:


model = ResNet50(weights='imagenet')

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224,224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# 将结果解码成元组列表（类，描述，概率）
#（批次中每个样品的一个这样的列表）
print('Predicted：', decode_predictions(preds, top=3)[0])
#Predicted： [('n02504458', 'African_elephant', 0.68021435), 
#('n01871265', 'tusker', 0.1594276),
#('n02504013', 'Indian_elephant', 0.1445787)]


# In[ ]:




