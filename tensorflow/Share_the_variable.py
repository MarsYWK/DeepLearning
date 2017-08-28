
# coding: utf-8

# In[50]:


import tensorflow as tf


# In[51]:


def conv_relu(input, kernel_shape, bias_shape):
    #创建名为weights的变量
    weights = tf.get_variable('weights', kernel_shape,
                             initializer = tf.random_normal_initializer())
    #创建名为biases的变量
    biases = tf.get_variable('biases', bias_shape,
                            initializer = tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input, weights,
                       strides=[1,1,1,1], padding='SAME')
    return tf.nn.relu(conv + biases)


# In[52]:


def my_image_filter(input_images):
    with tf.variable_scope('conv1'):
        #在这里创建的变量将被命名为'conv1/weights','conv1/biases'
        relu1 = conv_relu(input_images, [5,5,32,32], [32])
    with tf.variable_scope('conv2'):
        #在这里创建的变量将被命名为'conv2/weights','conv2/biases'
        return conv_relu(relu1, [5,5,32,32],[32])


# In[53]:


#通过reuse_variables()这个方法来指定共享
with tf.variable_scope('image_filters') as scope:
    result1 = my_image_filter(image1)
    scope.reuse_variables()
    result2 = my_image_filter(image2)


# In[54]:


#调用就会搜索一个已经存在的变量，
#他的全称和当前变量的作用域名+所提供的名字是否相等.
#如果不存在相应的变量，就会抛出ValueError 错误.
#如果变量找到了，就返回这个变量.
with tf.variable_scope('foo'):
    v = tf.get_variable('v',[1])
with tf.variable_scope('foo', reuse = True):
    v1 = tf.get_variable('v',[1])
assert v1 == v


# In[55]:


#变量作用域的主方法带有一个名称，它将会作为前缀用于变量名,
#并且带有一个重用标签来区分以上的两种情况.
#嵌套的作用域附加名字所用的规则和文件目录的规则很类似：
with tf.variable_scope('foo'):
    with tf.variable_scope('bar'):
        v = tf.get_variable('v',[1])
assert v.name == 'foo/bar/v:0'


# In[56]:


#注意你不能设置reuse标签为False.
#当前变量作用域可以用tf.get_variable_scope()进行检索并且reuse 标签可以通过调用
with tf.variable_scope('foo'):
    v = tf.get_variable('v',[1])
    tf.get_variable_scope().reuse_variables()
    v1 = tf.get_variable('v',[1])
assert v1 == v


# In[57]:


with tf.variable_scope('root'):
    #开始，作用域不重用
    assert tf.get_variable_scope().reuse == False
    with tf.variable_scope('foo'):
        #打开一个子范围，仍然不能重用。
        assert tf.get_variable_scope().reuse == False
    with tf.variable_scope('foo',reuse=True):
        #明确打开重用范围。
        assert tf.get_variable_scope().reuse == True
        with tf.variable_scope('bar'):
            #现在子范围继承了重用标志。
            assert tf.get_variable_scope().reuse == True
    #退出重用范围，返回到不重用的范围。
    assert tf.get_variable_scope().reuse == False


# In[58]:


#获取变量作用域
with tf.variable_scope('foo') as foo_scope:
    v = tf.get_variable('v',[1])
with tf.variable_scope(foo_scope):
    w = tf.get_variable('w',[1])
with tf.variable_scope('foo_scope',reuse=True):
    v1 = tf.get_variable('v',[1])
    w1 = tf.get_variable('w',[1])
assert v1 == v
assert w1 == w


# In[59]:


with tf.variable_scope('foo') as foo_scope:
    assert foo_scope.name == 'foo'
with tf.variable_scope('bar'):
    with tf.variable_scope('baz') as other_scope:
        assert other_scope.name == 'bar/baz'
        with tf.variable_scope(foo_scope) as foo_scope2:
            assert foo_scope2.name == 'foo' #不变


# In[60]:


# #变量作用域中的初始化器
with tf.variable_scope('foo', initializer=tf.constant_initializer(0.4)):
    v = tf.get_variable('v',[1])
    assert v.eval() == 0.4 #默认初始化器如上所述。
    w = tf.get_variable('w',[1],initializer=tf.constant_initializer(0.3))
    assert.w.eval() == 0.3 #特定的初始化器覆盖默认值。
    with tf.variable_scope('bar'):
        v = tf.get_variable('v',[1])
        assert v.eval() == o.4 #继承默认初始化
    with tf.variable_scope('baz',initializer=tf.constant_initializer(0.2)):
        v = tf.get_variable('v',[1])
        assert v.eval() == 0.2 #改变默认初始化器


# In[61]:


#在tf.variable_scope()中ops的名称
with tf.variable_scope('foo'):
    x = 1.0 + tf.get_variable('v',[1])
assert x.op.name == 'foo/add'


# In[ ]:


#名称作用域可以被开启并添加到一个变量作用域中,
#然后他们只会影响到ops的名称,而不会影响到变量.
with tf.variable_scope('foo'):
    with tf.name_scope('bar'):
        v = tf.get_variable('v',[1])
        x = 1.0 + v
assert v.name == 'foo/v:0'
assert x.op.name == 'foo/bar/add'

