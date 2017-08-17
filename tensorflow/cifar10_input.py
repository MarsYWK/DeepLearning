
# coding: utf-8

# In[2]:


import os


import tensorflow.python.platform
from six.moves import xrange
import tensorflow as tf

from tensorflow.python.platform import gfile


# In[3]:


#这个和原始CIFAR图像32x32不同
#如果改变这个数字，那么整个模型架构将会改变，任何模型都需要重新训练。
IMAGE_SIZE = 24

#描述CIFAR-10数据集的全局常量
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXANPLES_PER_EPOCH_FOR_EVAL = 10000


# In[5]:


def read_cifar10(filename_queue):
    
    class CIFAR10Record(object):
        pass
    result = CIFAR10Record()
    
    #在CIFAR-10数据集中图像维度
    label_bytes = 1 # 2是CIFAR-100
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    
    #每个记录都包含一个后跟图像的标签,有固定字节
    record_bytes = label_bytes + image_bytes
    #读取记录，从filename_queue获取文件名。 
    #没有CIFAR-10格式的页眉或页脚，
    #所以我们把header_bytes和footer_bytes保留为默认值0。
    reader = tf.FixedLengthRecordReader(record_bytes = record_bytes)
    result.key, value = reader.read(filename_queue)
    
    #从一个字符串转换为一个向量的uint8是record_bytes长
    record_bytes = tf.decode_raw(value, tf.uint8)
    
    #第一个字节表示从uint8-> int32转换的标签。
    result.label = tf.cast(
        tf.slice(record_bytes,[0],[label_bytes]), tf.int32)
    
    #标签后的剩余字节表示图像，
    #我们从[depth * height * width]重新形成为[depth，height，width]。
    depth_major = tf.reshape(tf.slice(record_bytes,[label_bytes],[image_bytes]),
                            [result.depth, result.height, result.width])
    #从[depth,height,width]转化为[height,width,depth]
    result.uint8image = tf.transpose(depth_major,[1,2,0])
    
    return result
    
    
    


# In[6]:


#构建排队的一批图像和标签
def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size):
    num_preprocess_threads = 16
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
    
    #显示图像
    tf.image_summary('images', images)
    return images, tf.reshape(label_batch, [batch_size])


# In[7]:


#使用Reader操作来构建CIFAR训练的失真输入。
def distorted_inputs(data_dir, batch_size):
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
               for i in xrange(1, 6)]
    for f in filenames:
        if not gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)
    
    #创建一个生成要读取的文件名的队列。
    filename_queue = tf.train.string_input_producer(filenames)
    
    #从文件名队列中的文件读取示例
    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)
    
    height = IMAGE_SIZE
    width = IMAGE_SIZE
    
    #随机裁剪图像的[height，width]部分。
    distorted_image = tf.image.random_crop(reshaped_image, [height, width])
    
    #随意地水平翻转图像。
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    
    #因为这些操作是不可交换的，所以考虑随机化它们的操作顺序。
    distorted_image = tf.image.random_brightness(distorted_image,
                                               max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                             lower=0.2, upper=1.8)
    
    #减去平均值，除以像素的方差。
    float_image = tf.image.per_image_whitening(distorted_image)
    
    #确保随机洗牌具有良好的混合性能。
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
    print('Filling queue with %d CIFAR images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)
    
    return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size)
    


# In[ ]:


#使用Reader操作构建CIFAR评估的输入
def inputs(eval_data, data_dir, batch_size):
    if not eval_data:
        filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                     for i in xrange(1, 6)]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        filenames = [os.path.join(data_dir, 'test_batch.bin')]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
        
    for f in filenames:
        if not gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)
    
    filename_queue = tf.train.string_input_producer(filenames)
    
    #从文件名队列中的文件读取示例。
    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)
    
    height = IMAGE_SIZE
    width = IMAGE_SIZ
    
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                         width, height)
    
    float_image = tf.image.per_image_whitening(resized_image)
    
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)
    
    return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size)
    
        

