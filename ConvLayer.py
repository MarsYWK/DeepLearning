
#卷积层初始化
class ConvLayer(object):
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

#计算卷积层输出的大小     
def calulate_output_size(input_size,filter_size, zero_padding,stride):
    return (input_size - filter_size +
            2* zero_padding) / stride +1
            
        
        
        
