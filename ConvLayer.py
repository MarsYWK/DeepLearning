import numpy as np


#激活函数
class ReluActivator(object):
    def forward(self, weighted_input):
        return max(0, weighted_input)
    
    #计算导数
    def backward(self,output):
        return 1 if output > 0 else 0
    

class ConvLayer(object):
    #卷积层初始化
    def __init__(self, input_width, input_height,
                 channel_number, filter_width,
                 filter_height, filter_number,
                 zero_padding, stride, activator,
                 learning_rate):
        self.input_width = input_width
        self.input_height = input_height
        self.channel_number = channel_number
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.filter_number = filter_number
        self.zero_padding = zero_padding
        self.stride = stride
        self.output_width = \
            ConvLayer.calculate_output_size(
                self.input_width,filter_width,zero_padding,
                stride)
        self.output_height = \
            ConvLayer.calculate_output_size(
                self.input_height,filter_height,zero_padding,
                stride)
        self.filters = []
        for i in range(filter_number):
            self.filters.append(Filter(filter_width,
                filter_height, self.channel_number))
        self.activator = activator
        self.learning_rate = learning_rate
        
        def forward(self,input_array):
            '''
            计算卷积层的输出
            输出结果保存在self.output_array
            '''
            self.input_array = input_array
            self.zero_padding_input_array = padding(input_array,self.zero_padding)
            for f in range(self.filter_number):
                filter = self.filters[f]
                conv(self.padded_input_array,
                     filter.get_weights(),self.output_array[f],
                     self.stride, filter.get_bias())
            element_wise_op(self.output_array,self.activator.forward)

        
        def bp_sensitivity_map(self,sensitivity_array,
                               activator):
            '''
            计算传递到上一层的sensitivity map
            sensitivity_array:本层的sensitivity map
            activator:上一层的激活函数
            '''
            #处理卷积步长，对原始sensitivity map进行扩展
            expanded_array = self.expand_sensitivity_map(
                sensitivity_array)
            
            #full卷积，对sensitivity map进行zero padding
            #虽然原始输入的zero padding单元也会获得残差
            #但是这个残差不需要继续向上传递，因此就不计算了
            expanded_width = expanded_array.shape[2]
            zp = (self.input_width +
                  self.filter_width -1 - expanded_width) / 2
            padded_array = padding(expanded_array, zp)
            #初始化delta_array,用于保存传递上一层的sensitivity map
            self.delta_array = self.create_delta_array()
            #对于具有多个filter的卷积层来说，最终传递到上一层的sensitivity map
            #相当于所有filter的sensitivity map纸盒
            for  f in range(self.filter_number):
                filter = self.filter[f]
                #讲filter权重翻转180度
                flipped_weights = np.array(map(
                    lambda i : np.rot90(i, 2),
                    filter.get_weights()))
                #计算与一个filter对应的delta_array
                delta_array = self.create_delta_array()
                for d in range(delta_array.shape[0]):
                    conv(padded_array[f], flipped_weights[d],
                         delta_array[d], 1, 0)
                self.delta_array += delta_array
            #讲计算结果与激活函数的偏导数做element-wise乘法操作
            derivate_array = np.array(self.input_array)
            element_wise_op(derivate_array,activator.backwar)
            self.delta_array *= derivate_array
        
        def expand_sensitivity_map(self, sensitivity_array):
            depth = sensitivity_array.shape[0]
            #确定扩展sensitivity map的大小
            #计算stride为1时sensitivity map的大小
            expanded_width = (self.input_width -
                              self.filter_width + 2 * self.zero_padding + 1)
            expanded_height = (self.input_height - 
                               self.filter_height + 2 * self.zero_padding + 1)
            #构建新的sensitivity_map
            expand_array = np.zeros((depth, expanded_height,
                                     expanded_width))
            #从原始sensitivity map拷贝误差值
            for i in range(self.output_height):
                for j in range(self.output_width):
                    i_pos = i * self.stride
                    j_pos = j * self.stride
                    expand_array[:,i_pos,j_pos] = sensitivity_array[:,i,j]
            
            return expand_array
        
        #创建保存传递到上一层的sensitivity map的数组
        def create_delta_array(self):
            return np.zeros((self.channel_number,
                             self.input_height, self.input_width))
        
        def bp_gradient(self, sensitivity_array):
            #处理卷积补偿，对原始sensitivity map进行扩展
            expanded_array = self.expand_sensitivity_map(sensitivity_array)
            for f in range(self.filter_number):
                #计算每个权重的梯度
                filter = self.filters[f]
                for d in range(filter.weights.shape[0]):
                    conv(self.padded_input_array[d],
                         expanded_array[f],
                         filter.weights_grad[d], 1, 0)
                #计算偏置项的梯度
                filter.bias_grad = expanded_array[f].sum()
        
        def update(self):
            '''
            按照梯度下降，更新权重
            '''
            for filter in self.fitlers:
                filter.update(self.learning_rate)

#对numpy数组进行element wise操作
def element_wise_op(array, op):
    for i in np.nditer(array, op_flags=['readwrite']):
        i[...] = op(i)
            
def conv(input_array,kernel_array,output_array,stride,bias):
    '''
    计算卷积，自动适配输入为2D和3D的情况
    '''
    channel_number = input_array.ndim
    output_width = output_array.shape[1]
    output_height = output_array.shape[0]
    kernel_width = kernel_array.shape[-1]
    kernel_height = kernel_array.shape[-2]
    for i in range(output_height):
        for j in range(output_width):
            output_array[i][j] = (
                get_patch(input_array, i, j, kernel_width,
                          kernel_height, stride) * kernel_array
                                  ).sum()+bias
                                       
def padding(input_array, zp):
    '''
    为数组增加Zero padding,自动适配输入为2D和3D的情况
    '''
    if zp == 0:
        return input_array
    else:
        if input_array.ndim == 3:
            input_width = input_array.shape[2]
            input_height = input_array.shape[1]
            input_depth = input_array.shape[0]
            padded_array = np.zeros((
                input_depth,
                input_height + 2 * zp,
                input_width + 2* zp))
            padded_array[:,
                         zp : zp + input_height,
                         zp : zp +input_width] = input_array
            return padded_array
        elif input_array.ndim == 2:
            input_width = input_array.shape[1]
            input_height = input_array.shape[0]
            padded_array = np.zeros((
                input_height + 2 * zp,
                input_width + 2 * zp))
            padded_array[zp : zp + input_height,
                         zp : zp +input_width] = input_array
            return padded_array
        
        
#计算卷积层输出的大小     
def calulate_output_size(input_size,filter_size, zero_padding,stride):
    return (input_size - filter_size +
            2* zero_padding) / stride +1

#Filter类保存了卷积层的参数以及梯度
class Filter(object):
    def __init__(self, width, height, depth):
        self.weights = np.random.uniform(-1e-4,1e-4,
                                         (depth, height, width))
        self.bias = 0
        self.weights_grad = np.zeros(
            self.weights.shape)
        self.bias_grad = 0
        
    def __repr__(self):
        return 'filter weights:\n%s\nbias:\n%s'%(
            repr(self.weights), repr(self.bias))
    
    def get_weights(self):
        return self.weights
    
    def get_bias(self):
        return self.bias
    
    #更新参数
    def update(self, learning_rate):
        self.weights -= learning_rate * self.weights_grad
        self.bias -= learning_rate * self.bias_grad





   
        
        
        