
# coding: utf-8

# In[13]:


import collections
import math
import os
import random
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  
import tensorflow as tf


# In[14]:


#步骤1，下载数据
url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
    #如果不存在，请下载文件，并确保其大小正确
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify '+ filename + '.Can you get to it with a browser?')
    return filename

filename = maybe_download('text8.zip', 31344016)


# In[15]:


#将数据读入字符串列表。
def read_data(filename):
    '''提取包含在zip文件中的第一个文件作为单词列表。'''
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

vocabulary = read_data(filename)
print('Data size', len(vocabulary))


# In[16]:


#步骤2：构建字典并使用UNK替换罕见单词
vocabulary_size = 50000

def build_dataset(words, n_words):
    '''将原始输入处理为数据集'''
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words -1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0: #dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(),dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

data, count, dictionary, reversed_dictionary = build_dataset(vocabulary,
                                                             vocabulary_size)

del vocabulary #减少内存

print('Most common words (+UNK)',count[:5])
print('Sample data', data[:10], [reversed_dictionary[i] for i in data[:10]])

data_index = 0
    
    


# In[17]:


#步骤3：生成skip-gram 模型的训练批次的功能。
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size),dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window +1 # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        random.shuffle(context_words)
        words_to_use = collections.deque(context_words)
        for j in range(num_skips):
            batch[i * num_skips + j] = buffer[skip_window]
            context_word = words_to_use.pop()
            labels[i * num_skips +j ,0] = buffer[context_word]
        if data_index == len(data):
            buffer[:] = data[:span]
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
        
    #回溯一点点，以避免在批次结束时跳过单词
    data_index = (data_index + len(data) - span ) % len(data)
    return batch,labels

batch, labels = generate_batch(batch_size=8,num_skips=2,skip_window=1)
for i in range(8):
    print(batch[i], reversed_dictionary[batch[i]],
         '->', labels[i, 0], reversed_dictionary[labels[i,0]])


# In[18]:


#步骤4，构建并训练skip-gram模型

batch_size = 128
embedding_size = 128
skip_window = 1 #左右要考虑多少个单词。
num_skips = 2 #＃多次重用输入生成标签。



#我们选择一个随机验证集来采样最近的邻居。
#在这里，我们将验证样本限制为具有低数字ID的单词，通过构造也是最常见的单词。
valid_size = 16 #随机集合的词来评估相似度。
valid_window = 100 
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64 #抽样的负样本数。

graph = tf.Graph()

with graph.as_default():
    
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    
    #由于缺少GPU实现，操作和变量固定在CPU上
    with tf.device('/cpu:0'):
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        
    #构造NCE损失的变量
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                           stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
    
    #计算批次的平均NCE损失。 
    #每次评估损失时，tf.nce_loss会自动绘制负标签的新样本。
    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                      biases = nce_biases,
                      labels = train_labels,
                      inputs = embed,
                      num_sampled = num_sampled,
                      num_classes=vocabulary_size))
    
    
    #使用1.0的学习速率构建SGD优化器。
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
    
    #计算minibatch示例和所有嵌入之间的余弦相似度。
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings),1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dataset)
    similarity = tf.matmul(
        valid_embeddings, normalized_embeddings, transpose_b=True)
    
    #变量初始化
    init = tf.global_variables_initializer()
    


# In[19]:


#步骤5，开始训练
num_steps = 100001

with tf.Session(graph=graph) as session:
    #在使用它们之前，我们必须初始化所有变量。
    init.run()
    print('Initialized')
    
    average_loss = 0
    for step in xrange(num_steps):
        batch_inputs, batch_labels = generate_batch(
            batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs,
                    train_labels: batch_labels}
        
        #我们通过评估优化器操作
        #（包括在session.run（）的返回值列表中执行一个更新步骤）
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val
        
        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            #平均亏损是过去2000批次损失的估计。
            print('Average loss at step', step, ':', average_loss)
            average_loss = 0
        
        #请注意，这是昂贵的（如果每500步计算一次约20％的减速）
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in xrange(valid_size):
                valid_word = reversed_dictionary[valid_examples[i]]
                top_k = 8 
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = 'Nearest to %s:' % valid_word
                for k in xrange(top_k):
                    close_word = reversed_dictionary[nearest[k]]
                    log_str = '%s %s,'% (log_str, close_word)
                print(log_str)
    final_embeddings = normalized_embeddings.eval()                
        


# In[21]:


#步骤6：可视化

#pylint: disable = missing-docstring
#绘制嵌入距离的可视化功能。
def plot_with_labels(low_dim_embs, labels, filename):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                    xy=(x, y),
                    xytext=(5,2),
                    textcoords='offset points',
                    ha='right',
                    va='bottom')
    plt.savefig(filename)
    
try:
    #pyint: disable=g-import-not-at-top
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    
    tsne = TSNE(perplexity=30, n_components=2,
               init='pca',n_iter=5000,method='exact')
    plot_only = 500
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [reversed_dictionary[i] for i in xrange(plot_only)]
    plot_with_labels(low_dim_embs, labels, os.path.join('', 'tsne.png'))
    
except ImportError as ex:
    print('Please install sklearn, matplotlib, and scipy to show embeddings.')
    print(ex)


# In[ ]:




