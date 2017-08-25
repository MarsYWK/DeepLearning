
# coding: utf-8

# In[55]:


import tensorflow as tf
import numpy as np

#导入可视化需要的库
import PIL.Image
from io import BytesIO
from IPython.display import clear_output, Image, display


# In[56]:


#一个用于表示池塘表面状态的函数。
def DisplayArray(a, fmt='jpeg', rng=[0,1]):
    #将数组显示为图片
    a = (a - rng[0]) / float(rng[1] - rng[0]) * 255
    a = np.uint8(np.clip(a, 0, 255))
    f = BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    clear_output(wait = True)
    display(Image(data = f.getvalue()))


# In[57]:


#打开一个 TensorFlow 的交互会话
sess = tf.InteractiveSession()


# In[58]:


#定义计算函数
def make_kernel(a):
    #将2D数组转换为卷积内核
    a = np.asarray(a)
    a = a.reshape(list(a.shape) + [1, 1])
    return tf.constant(a, dtype=1)

def simple_conv(x, k):
    #简化2D卷积运算
    x = tf.expand_dims(tf.expand_dims(x, 0), -1)
    y = tf.nn.depthwise_conv2d(x, k [1,1,1,1], padding='SAME')
    return y[0, :, :, 0]

def laplace(x):
    #计算一个数组的二维拉普拉斯算子
    laplace_k = make_kernel([[0.5, 1.0, 0.5],
                            [1.0, -6, 1.0],
                            [0.5, 1.0, 0.5]])
    return simple_conv(x, laplace_k)


# In[59]:


#定义偏微分方程
#首先,我们需要创建一个完美的 500 × 500 的正方形池塘,就像是我们在现实中找到的一样。
N= 500

#然后，我们需要创建了一个池塘和几滴将要坠入池塘的雨滴。
u_init = np.zeros([N,N],dtype=np.float32)
ut_init = np.zeros([N,N],dtype=np.float32)

#有些雨滴随机点击池塘
for n in range(40):
    a,b = np.random.randint(0, N, 2)
    u_init[a,b] = np.random.uniform()


# In[60]:


#指定该微分方程的一些详细参数
eps = tf.placeholder(tf.float32, shape=())
damping = tf.placeholder(tf.float32, shape=())

#创建模拟状态的变量
U = tf.Variable(u_init)
Ut = tf.Variable(ut_init)

#离散PDE更新规则
U_ = U + eps * Ut
Ut_ = Ut + eps * (laplace(U) - damping * Ut)

#更新状态
step = tf.group(
    U.assign(U_),
    Ut.assign(Ut_))



# In[ ]:


#开始仿真
tf.global_variables_initializer().run()

for i in range(1000):
    step.run({eps: 0.03, damping: 0.04})
    DisplayArray(U.eval(), rng=[-0.1, 0.1])


# In[ ]:




