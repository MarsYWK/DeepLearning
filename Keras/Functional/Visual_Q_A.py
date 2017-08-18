
# coding: utf-8

# In[22]:


from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model, Sequential


# In[23]:


#首先，我们使用Sequential模型定义一个视觉模型。
#这个模型会将图像编码成一个向量。
vision_model = Sequential()
vision_model.add(Conv2D(64, (3,3), activation='relu',
                        padding='same',input_shape=(3,224,224)))
vision_model.add(Conv2D(64, (3,3), activation='relu'))
vision_model.add(MaxPooling2D((2,2)))
vision_model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
vision_model.add(Conv2D(128, (3,3), activation='relu'))
vision_model.add(MaxPooling2D((2,2)))
vision_model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
vision_model.add(Conv2D(256, (3,3), activation='relu'))
vision_model.add(Conv2D(256, (3,3), activation='relu'))
vision_model.add(MaxPooling2D((2,2)))
vision_model.add(Flatten())

#现在我们来看看我们的视觉模型的输出：
image_input = Input(shape=(3, 224, 224))
encoded_image = vision_model(images_input)

#＃接下来，我们定义一个语言模型来将问题编码成一个向量。
#每个问题至多100个字，
#，我们将索引单词作为整数从1到9999。
question_input = Input(shape=(100,), dtype='int32')
embedded_question = Embedding(input_dim=10000, output_dim=256, input_length=100)(question_input)
encoded_question = LSTM(256)(embedded_question)

#让我们连接问题向量和图像向量：
merged = keras.layers.concatenate([encoded_question, encoded_image])

#让我们训练一个超过1000个单词的逻辑回归：
output = Dense(1000, activation='softmax')(merged)

vqa_model = Model(inputs=[image_input, question_input], outputs=output)

#下一阶段将对实际数据进行培训。


# In[ ]:




