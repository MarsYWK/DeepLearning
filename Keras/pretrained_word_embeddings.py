
# coding: utf-8

# In[ ]:


from __future__ import print_function

import os

import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model


# In[ ]:


BASE_DIR = 'word_embeddings_data'
GLOVE_DIR = BASE_DIR + '/glove.6B/'
TEXT_DATA_DIR = BASE_DIR + '/20_newsgroup/'
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2


# In[ ]:


#首先，将嵌入中的索引映射字设置为其嵌入向量
print('Indexing word vectors.')

embeddings_index = {}
fpath = os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')

f = open(fpath,'rb')

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()


print('Found %s word vectors.'%len(embeddings_index))



# In[ ]:


#第二，准备文本样本及其标签
print('Processiong text dataset')

texts = [] #文本样本列表
labels_index = {} #字典将标签名称映射到数字标识
labels = [] #标签id列表
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            fpath = os.path.join(path, fname)
            if sys.version_info < (3,):
                f = open(fpath)
            else:
                f = open(fpath, encoding='latin-1')
            t = f.read()
            i = t.find('\n\n') #跳过标题
            if 0 < i:
                t = t[i:]
            texts.append(t)
            f.close()
            labels.append(label_id)

print('Found %s texts.' % len(texts))


# In[ ]:


#最后，将文本样本矢量化为2D整数张量
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)


# In[ ]:


#将数据分成训练集和验证集
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

print('Preparing embedding matrix.')


# In[ ]:


#准备嵌入矩阵
num_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        #嵌入索引中找不到的字将全为零。
        embedding_matrix[i] = embedding_vector
        
        
#将预先训练好的词嵌入到嵌入层中
#注意到我们设置trainable = False以保持嵌入的固定
embedding_layer = Embedding(num_words,
                           EMBEDDING_DIM,
                           weights=[embedding_matrix],
                           input_length = MAX_SEQUENCE_LENGTH,
                           trainable=False)


# In[ ]:


print('Training model.')

#用一个全局maxpooling来训练一个1D convnet
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,),dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(35)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer = 'rmsprop',
              metrics=['acc'])

model.fit(x_train, y_train,
         batch_size=128,
         epochs = 10,
         validation_data=(x_val, y_val))

