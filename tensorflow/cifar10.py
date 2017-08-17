
# coding: utf-8

# In[2]:


import gzip
import os
import re
import sys
import tarfile

import tensorflow.python.platform
from six.moves import xrange
import tensorflow as tf

import cifar10_input


# In[3]:


FLAGS = tf.app.flags.FLAGS

#基础模型参数
tf.app.flags.DEFINE_integer('batch_size',128,
                           '''Number of images to process in a batch.''')
tf.app.flags.DEFINE_string('data_dir','cifar10_data/',
                          '''Path to the CIFAR-10 data directory.''')

#描述CIFAR-10数据集的全局常量。
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

#描述训练过程的常数
MOVING_AVERAGE_DECAY = 0.9999  #用于移动平均的衰减。
NUM_EPOCHS_PER_DECAY = 350.0  #学习率衰减后的训练次数
LEARNING_RATE_DECAY_FACTOR = 0.1  #学习率衰减因子
INITIAL_LEARNING_RATE = 0.1  #初始化学习率

#如果一个模型训练有多个GPU的前缀，所有的操作名称与tower名称区分操作。 
#请注意，当可视化模型时，该前缀将从摘要的名称中删除。
TOWER_NAME = 'tower'

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


# In[3]:


#为激活创建摘要
def _activation_summary(x):
    #从名称中删除'tower_ [0-9] /'，以防这是一个多GPU训练会话。
    #这有助于在tensorboard上呈现的清晰度。
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


# In[5]:


#创建存储在CPU内存中的变量
def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


# In[6]:


#创建具有权重衰减的初始化变量。
def _variable_with_weight_decay(name, shape, stddev, wd):
    var = _variable_on_cpu(name, shape,
                         tf.truncated_normal_initializer(stddev=stddev))
    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var),wd,name='weight_loss')
        tf.add_to_collection('losses',weight_decay)
    return var


# In[7]:


#使用Reader操作来构建CIFAR训练的失真输入
def distorted_inputs():
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir,'cifar-10-batches-bin')
    return cifar10_input.distorted_inputs(data_dir=data_dir,
                                         batch_size=FLAGS.batch_size)


# In[8]:


#使用Reader操作构建CIFAR评估的输入
def inputs(eval_data):
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir,'cifar-10-batches-bin')
    return cifar10-input.inputs(eval_data=eval_data, data_dir=data_dir,
                               batch_size = FLAGS.batch_size)


# In[9]:


#建立CIFAR-10模型
def inference(images):
    #我们使用tf.get_variable（）而不是tf.Variable（）来实例化所有变量，
    #以便在多个GPU训练运行之间共享变量。如果我们仅在单个GPU上运行此模型，
    #则可以通过替换tf的所有实例来简化此函数 .用tf.Variable（）替换tf.get_variable()
    
    #conv1
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights',shape=[5,5,3,64],
                                            stddev=1e-4,wd=0.0)
        conv = tf.nn.conv2d(images,kernel,[1,1,1,1],padding='SAME')
        biases = _variable_on_cpu('biases',[64],tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias,name=scope.name)
        _activation_summary(conv1)
    
    #pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1],strides=[1,2,2,1],
                          padding='SAME',name='pool1')
    
    
    #norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                     name='norm1')
    
    #conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights',shape=[5,5,64,64],
                                            stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(norm1, kernel, [1,1,1,1], padding='SAME')
        biases = _variable_on_cpu('biases',[64],tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv2)
    
    #norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 /9.0, beta=0.75,
                     name='norm2')
    
    #pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1,3,3,1],
                          strides=[1,2,2,1], padding='SAME')
    
    #local3
    with tf.variable_scope('local3') as scope:
        #一维，以便我们可以执行单个矩阵乘法。
        dim = 1
        for d in pool2.get_shape()[1:].as_list():
            dim *= d
        reshape = tf.reshape(pool2, [FLAGS.batch_size, dim])
        
        weights = _variable_with_weight_decay('weights',shape=[dim, 384],
                                             stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases , name=scope.anme)
        _activation_summary(local3)
    
    #local4
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay('weights',shape=[384,192],
                                             stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases',[192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
        _activation_summary(local4)
        
    #softmax, i.e. softmax(WX + b)
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights',[192,NUM_CLASSES],
                                             stddev=1/192, wd=0.0)
        biases = _variable_on_cpu('biases',[NUM_CLASSES],tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
        _activation_summary(softmax_linear)
    
    return softmax_linear
    


# In[10]:


#将L2Loss添加到所有可训练的变量中
def loss(logits, labels):
    
    sparse_labels = tf.reshape(labels, [FLAGS.batch, 1])
    
    indices = tf.reshape(range(FLAGS.batch_size),[FLAGS.batch_size, 1])
    concated = tf.concat(1, [indices, sparse_labels])
    dense_labels = tf.sparse_to_dense(concated,
                                     [FLAGS.batch_size, NUM_CLASSES],
                                     1.0, 0.0)
    #计算批次间的平均交叉熵损失
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits, dense_labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    
    #总损失定义为交叉熵损失加上所有的权重衰减项（L2损失）
    return tf.add_n(tf.get_collection('losses'),name='total_loss')



# In[11]:


#在CIFAR-10模型中增加损失汇总。
def _add_loss_summaries(total_loss):
    
    #计算所有单个损失的移动平均值和总损失
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    
    #对所有个人损失和总损失附加标量汇总; 
    #对平均版本的损失做同样的事情。
    for l in losses + [total_loss]:
        #将每个损失命名为“（原始”），并将损失的移动平均版本命名为原始损失名称。
        tf.scalar_summary(l.op.name +' (raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))
        
    return loss_averages_op


# In[13]:


#训练模型
def train(total_loss, global_step):
    #影响学习率的变量
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
    
    #基于步数，以指数方式衰减学习率。
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                   global_step,
                                   decay_steps,
                                   LEARNING_RATE_DECAY_FACTOR,
                                   staircase=True)
    tf.scalar_summary('learning_rate',lr)
    
    #生成所有损失和相关摘要的移动平均值。
    loss_averages_op = _add_loss_summaries(total_loss)
    
    #计算梯度
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)
        
    #应用梯度
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    
    #添加可修改变量的直方图
    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)
    
    #添加梯度的直方图
    for grad, var in grads:
        if grad:
            tf.histogram_summary(var.op.name + '/gradients', grad)
    
    #跟踪所有可训练变量的移动平均值。
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op

    
    
    


# In[ ]:


def maybe_download_and_extract():
    dest_directory = FLAGS.data_dir
    if not os.path.exist(dest_directory):
        os.mkdir(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>>Downloading %s %.1f%%'%(filename,
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath,
                                                reporthook=_progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath,'r:gz').extractall(dest_directory)

