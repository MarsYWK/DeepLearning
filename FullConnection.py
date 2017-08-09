import numpy as np


#定义激活函数：sigmoid函数
def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1.0 - np.tanh(x)*np.tanh(x)

def logistic(x):
    return 1/(1+np.exp(-x))

def logistic_deriv(x):
    return logistic(x)*(1-logistic(x))

#定义神经网络结构
class FullConnection:
    def __init__(self,layers,activation = 'tanh'):
        '''
        参数layers:包含每个层中节点数的列表应至少为两个值
        参数activation：激活函数 默认为'tanh'
        '''
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_deriv
        elif activation =='tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv
        
        #初始化权重
        self.weights = []
        for i in range(1,len(layers)-1):
            self.weights.append((2*np.random.random((layers[i-1]+1,layers[i]+1))-1)*0.25)
            self.weights.append((2*np.random.random((layers[i]+1,layers[i+1]))-1)*0.25)
  
    #建立模型          
    def fit(self,X,y,learning_rate=0.2,epochs = 10000):
        X = np.atleast_2d(X)
        temp = np.ones([X.shape[0],X.shape[1]+1])
        #增加偏向
        temp[:,0:-1] = X
        X = temp
        y = np.array(y)
    
        #一次训练循环
        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]]
        
            #向前传播
            for l in range(len(self.weights)):
                a.append(self.activation(np.dot(a[l],self.weights[l])))
                #计算输出层的误差
                error = y[i] - a[-1]
                deltas = [error * self.activation_deriv(a[-1])]
        
            #反向传播
            for l in range(len(a) - 2, 0 ,-1):
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_deriv(a[l]))
                deltas.reverse()
                for i in range(len(self.weights)):
                    layer = np.atleast_2d(a[i])
                    delta = np.atleast_2d(deltas[i])
                    self.weights[i] += learning_rate * layer.T.dot(delta)
    
    #进行预测
    def predit(self,X):
        X = np.array(X)
        temp = np.ones(X.shape[0]+1)
        temp[0:-1] = X
        a = temp
        for l in range(0,len(self.weights)):
            a = self.activation(np.dot(a,self.weights[l]))
        return a
        
#进行测试
nn = FullConnection([3,2,1],'tanh')
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])
nn.fit(X,y)
for i in [[0,0],[0,1],[1,0],[1,1]]:
    print(i,nn.predit(i)) 
            

