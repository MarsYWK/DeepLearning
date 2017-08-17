
# coding: utf-8

# In[1]:


import math
import tensorflow as tf


# In[2]:


NUM_CLASSES = 10
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE


# In[4]:


def inference(images, hidden1_units, hidden2_units):
    #建立MNIST模型取决于在推断中可能用的位置
    
    #Hidden 1
    with tf.name_scope('hidden1'):
        weights = tf.Variable(
            tf.truncated_normal([IMAGE_PIXELS,hidden1_units],
                               stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden1_units]),
                            name='biases')
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)

    #Hidden 2
    with tf.name_scope('hidden2'):
        weights = tf.Variable(
            tf.truncated_normal([hidden1_units,hidden2_units],
                               stddev=1.0 / math.sqrt(float(hidden1_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden2_units]),
                            name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
    
    #Linear
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(
            tf.truncated_normal([hidden2_units, NUM_CLASSES],
                               stddev=1.0 / math.sqrt(float(hidden2_units))),
            name='weights')
        biase = tf.Variable(tf.zeros([NUM_CLASSES]),
                           name='biases')
        logits = tf.matmul(hidden2, weights) + biases
    return logits
    


# In[5]:


def loss(logits, labels):
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels = labels, logits=logits, name='xentropy')
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')
    
    


# In[6]:


def training(loss, learning_rate):
    tf.summary.scalar('loss',loss)
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    #创建一个变量来跟踪全局步骤
    global_step = tf.Variable(0, name='global_step',trainable=False)
    
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op
    
    


# In[ ]:


def evaluation(logits, labels):
    #对于分类器模型，我们可以使用in_top_k Op.
    #It返回一个布尔张量，其形状为[batch_size]，
    #对于该示例的所有logtis的标签位于顶部k（这里为k = 1））的示例，
    #该值为true。
    correct = tf.nn.in_top_k(logits, labels, 1)
    #返回正确的个数
    return tf.reduce_sum(tf.cast(correct, tf.int32))

