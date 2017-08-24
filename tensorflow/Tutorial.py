
# coding: utf-8

# In[83]:


# encoding:utf-8


import tensorflow as tf


# 建立模型

x=tf.placeholder(tf.float32, [None, 28 * 28])
y=tf.placeholder(tf.float32, [None, 10])

w=tf.Variable(tf.zeros([28*28,10]))
b=tf.Variable(tf.zeros([10]))
a=tf.nn.softmax(tf.matmul(x, w) + b)

cross_entropy=tf.reduce_mean(-tf.reduce_sum(y * tf.log(a), reduction_indices=[1]))
optimizer=tf.train.GradientDescentOptimizer(0.5)
train_step=optimizer.minimize(cross_entropy)

correct_prediction=tf.equal(tf.argmax(a,1), tf.argmax(y, 1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


# 输入数据，调用模型


from tensorflow.examples.tutorials.mnist import input_data
from time import time

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)   # 读取数据集

session=tf.InteractiveSession()
tf.global_variables_initializer().run()

fetches={
    'step':train_step,
    'intermediate_accuracy':accuracy
}
begin_time=time()
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)    # 获得一批100个数据
    train_step.run({x: batch_xs, y: batch_ys})   # 给训练模型提供输入和输出
    # session.run(train_step, {x: batch_xs, y: batch_ys}) # 和上面这句是等效的

    # 如果想要把模型的中间结果输出看看，使用方法一。
    # 方法一：fetches为想要查看的值，已经在外部定义。此方法在我的机器上耗时7.5s
    # vals=session.run(fetches, {x: batch_xs, y: batch_ys}) # 和上面这句是等效的
    # intermediate_accuracy=vals['intermediate_accuracy']

    # 方法二：分别run各值。这种方法在我的机器上耗时35s，而且在很多情况下会导致model不能正常运行(我学习时遇到的大坑之一)。
    # session.run(train_step, {x: batch_xs, y: batch_ys}) # 和上面这句是等效的
    # ans=session.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
    
    
print(session.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels}))
print(time()-begin_time)

