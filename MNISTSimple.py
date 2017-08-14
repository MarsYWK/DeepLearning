
# coding: utf-8

# In[28]:


import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data


# In[29]:


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#x是一个占位符placeholder,None表示此张量的第一个维度是任何长度的
x = tf.placeholder(tf.float32, [None, 784])

#用全为零的张量初始化W和b
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W) + b)

y_ = tf.placeholder("float", [None,10])

#计算交叉熵
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

#梯度下降算法以0.01的学习速率最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#初始化我们创建的变量
init = tf.initialize_all_variables()

#在一个Session里面启动我们的模型,，并且初始化变量
sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    
#tf.argmax能给出某个tensor对象在某一维上的其数据最大值所在的索引值
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

