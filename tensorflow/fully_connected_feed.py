
# coding: utf-8

# In[8]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time


from six.moves import xrange
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist


# In[9]:


#基本型号参数作为外部标志
FLAGS = None


# In[10]:


#生成占位符变量来表示输入张量
def placeholder_inputs(batch_size):
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         mnist.IMAGE_PIXELS))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    return images_placeholder, labels_placeholder
    
def fill_feed_dict(data_set,images_pl,labels_pl):
    #填充feed_dict以训练给定的步骤
    images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size,
                                                 FLAGS.fake_data)
    feed_dict={
        images_pl:images_feed,
        labels_pl:labels_feed,
    }
    return feed_dict


# In[11]:


#评估
def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set):
    true_count = 0 #计算正确的预测数
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size
    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set,
                                  images_placeholder,
                                  labels_placeholder)
        true_count += sess.run(eval_correct,feed_dict=feed_dict)
    precision = float(true_count) / num_examples
    print('Num examples: %d Num correct: %d Precision @1: %0.04f'%
         (num_examples,true_count,precision))


# In[12]:


def run_training():
    data_sets = input_data.read_data_sets(FLAGS.input_data_dir,FLAGS.fake_data)
    
    #告诉TensorFlow将该模型内置到默认图形中
    with tf.Graph().as_default():
        images_placeholder, labels_placeholder = placeholder_inputs(
        FLAGS.batch_size)
        
        #构建一个从推理模型计算预测的图表
        logits = mnist.inference(images_placeholder,
                                FLAGS.hidden1,
                                FLAGS.hidden2)
        #添加到图表中用于损失函数计算
        loss = mnist.loss(logits, labels_placeholder)
        
        #添加到图表计算和应用梯度的操作
        train_op = mnist.training(loss,FLAGS.learning_rate)
        
        #添加Op，以便在评估过程中将logit与标签进行比较。
        eval_correct = mnist.evaluation(logits,labels_placeholder)
        
        #根据汇总的TF收集构建摘要Tensor
        summary = tf.summary.merge_all()
        
        #变量初始化
        init = tf.global_variables_initializer()
        
        #创建一个保存程序来编写训练检查点
        saver = tf.train.Saver()
        
        #在图上建立会话
        sess = tf.Session()
        
        #实例化一个SummaryWriter以输出摘要和Graph
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir,sess.graph)
        
        sess.run(init)
        
        #开始循环训练
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            
            feed_dict = fill_feed_dict(data_sets.train,
                                      images_placeholder,
                                      labels_placeholder)
            
            _,loss_value = sess.run([train_op,loss],
                                   feed_dict=feed_dict)
            
            duration = time.time() - start_time
            
            #撰写摘要，并经常打印概述
            if step % 100 == 0:
                print('Step %d：loss = %.0f(%.3f sec)'%(
                    step,loss_value,duration))
                
                #更新事件文件
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()
            
            #保存检查点并定期评估模型
            if(step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
                saver.save(sess,checkpoint_file,global_step=step)
                
                 #评估训练集
                print('Training Data Eval:')
                do_eval(sess,
                        eval_correct,
                       images_placeholder,
                       labels_placeholder,
                       data_sets.train)
                
                #评估验证集
                print('Validation Data Eval:')
                do_eval(sess,
                        eval_correct,
                       images_placeholder,
                       labels_placeholder,
                       data_sets.validation)
                #评估测试集
                print('Test Data Eval:')
                do_eval(sess,
                        eval_correct,
                       images_placeholder,
                       labels_placeholder,
                       data_sets.test)
        
        


# In[13]:


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    run_training()


# In[14]:



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=2000,
        help='Number of steps to run trainer.'
    )
    parser.add_argument(
        '--hidden1',
        type=int,
        default=128,
        help='Number of units in hidden layer 1.'
    )
    parser.add_argument(
        '--hidden2',
        type=int,
        default=32,
        help='Number of units in hidden layer 2.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Batch size. Must divide evenly into the dataset sizes'
    )
    parser.add_argument(
        '--input_data_dir',
        type=str,
        default=os.path.join(os.getenv('TEST_TMPDIR','/tmp'),
                                      'tensorflow/mnist/input_data'),
        help='Directory to put the input data.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default=os.path.join(os.getenv('TEST_TEPDIR','/tmp'),
                            'tensorflow/mnist/logs/fully_connected_feed'),
        help='directory to put the log data'
    )
    parser.add_argument(
        '--fake_data',
        default=False,
        help='If true, uses fake data for unit testing.',
        action='store_true'
    )
    
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

    
    

