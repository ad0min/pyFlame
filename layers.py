import numpy as np
from activation_function import ReLU, Sigmoid, TanH, Softmax
import math

class Layer(object):
	def set_input_shape(self, shape):
		self.input_shape = shape 

	def get_layer_name(self):
		return self.__class__.__name__

	def forward_pass(self, X, training):
		raise NotImplementedError()

	def backward_pass(self, accum_grad):
		raise NotImplementedError()

	def output_shape(self):
		raise NotImplementedError()

class Conv2D(Layer):

	def __init__(self, n_filters, filter_shape, input_shape = None, padding = 0, stride = 1):
		self.n_filters = n_filters
		self.filter_shape = filter_shape
		self.input_shape = input_shape
		self.padding = padding
		self.stride = stride 

	def initialize(self):
		filter_height, filter_width = self.filter_shape
		channels = self.input_shape[0]
		self.W = np.random.normal(0, 0.001, size = (self.n_filters, channels, filter_height, filter_width))
		self.bias = np.zeros((self.n_filters, 1))

	def forward_pass(self, X, training = True):
		batch_size, channels, height, width = X.shape 
		self.layer_input = X 
		self.X_col = image_to_column(X, self.filter_shape, stride = self.stride, output_shape = self.padding)
		self.W_col = self.W.reshape((self.n_filters, -1))
		# print(self.X_col.shape, X.shape, self.W_col.shape, self.input_shape, self.W.shape, self.n_filters)
		output = np.dot(self.W_col, self.X_col) + self.bias
		# print(output.shape, self.output_shape())
		output = output.reshape(self.output_shape() + (batch_size, ))
		return output.transpose(3,0,1,2)

	def backward_pass(self, accum_grad):
		accum_grad = accum_grad.transpose(1,2,3,0).reshape(self.n_filters, -1)

		grad_w = np.dot(accum_grad, self.X_col.T).reshape(self.W.shape)
		grad_bias = np.sum(accum_grad, axis= 1, keepdims= True)
		self.W -= grad_w
		self.bias -= grad_bias

		accum_grad = np.dot(self.W_col.T, accum_grad)
		accum_grad = column_to_image(accum_grad, self.layer_input.shape, self.filter_shape, stride = self.stride, output_shape= self.padding)
		return accum_grad

	def output_shape(self):
		channels, height, width = self.input_shape
		pad_h, pad_w = determine_padding(self.filter_shape, output_shape= self.padding)
		output_height = (height + np.sum(pad_h) - self.filter_shape[0]) / self.stride + 1
		output_width = (width + np.sum(pad_w) - self.filter_shape[1]) / self.stride + 1
		return self.n_filters, int(output_height), int(output_width)


activation_functions = {
    'relu': ReLU,
    'sigmoid': Sigmoid,
    'softmax': Softmax,
    'tanh': TanH,
}

class Activation(Layer):

    def __init__(self, name):
        self.activation_name = name
        self.activation_func = activation_functions[name]()
        self.trainable = True

    def layer_name(self):
        return "Activation (%s)" % (self.activation_func.__class__.__name__)

    def forward_pass(self, X, training=True):
        # print("forward: Activate", self.activation_name, X.shape)
        self.layer_input = X
        return self.activation_func(X)

    def backward_pass(self, accum_grad):
        # print("backward: Activate", self.activation_name, self.layer_input.shape, accum_grad.shape)
        return accum_grad * self.activation_func.gradient(self.layer_input)

    def output_shape(self):
        return self.input_shape


def determine_padding(filter_shape, output_shape="same"):

    # No padding
    if output_shape == "valid":
        return (0, 0), (0, 0)
    # Pad so that the output shape is the same as input shape (given that stride=1)
    elif output_shape == "same":
        filter_height, filter_width = filter_shape

        # Derived from:
        # output_height = (height + pad_h - filter_height) / stride + 1
        # In this case output_height = height and stride = 1. This gives the
        # expression for the padding below.
        pad_h1 = int(math.floor((filter_height - 1)/2))
        pad_h2 = int(math.ceil((filter_height - 1)/2))
        pad_w1 = int(math.floor((filter_width - 1)/2))
        pad_w2 = int(math.ceil((filter_width - 1)/2))
        

        return (pad_h1, pad_h2), (pad_w1, pad_w2)


# Reference: CS231n Stanford
def get_im2col_indices(images_shape, filter_shape, padding, stride=1):
    # First figure out what the size of the output should be
    batch_size, channels, height, width = images_shape
    filter_height, filter_width = filter_shape
    pad_h, pad_w = padding
    out_height = int((height + np.sum(pad_h) - filter_height) / stride + 1)
    out_width = int((width + np.sum(pad_w) - filter_width) / stride + 1)

    i0 = np.repeat(np.arange(filter_height), filter_width)
    i0 = np.tile(i0, channels)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(filter_width), filter_height * channels)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(channels), filter_height * filter_width).reshape(-1, 1)

    return (k, i, j)

def image_to_column(images, filter_shape, stride, output_shape='same'):
    filter_height, filter_width = filter_shape

    pad_h, pad_w = determine_padding(filter_shape, output_shape)

    # Add padding to the image
    images_padded = np.pad(images, ((0, 0), (0, 0), pad_h, pad_w), mode='constant')

    # Calculate the indices where the dot products are to be applied between weights
    # and the image
    k, i, j = get_im2col_indices(images.shape, filter_shape, (pad_h, pad_w), stride)

    # Get content from image at those indices
    cols = images_padded[:, k, i, j]
    channels = images.shape[1]
    # Reshape content into column shape
    cols = cols.transpose(1, 2, 0).reshape(filter_height * filter_width * channels, -1)
    return cols



# Method which turns the column shaped input to image shape.
# Used during the backward pass.
# Reference: CS231n Stanford
def column_to_image(cols, images_shape, filter_shape, stride, output_shape='same'):
    batch_size, channels, height, width = images_shape
    pad_h, pad_w = determine_padding(filter_shape, output_shape)
    height_padded = height + np.sum(pad_h)
    width_padded = width + np.sum(pad_w)
    images_padded = np.empty((batch_size, channels, height_padded, width_padded))

    # Calculate the indices where the dot products are applied between weights
    # and the image
    k, i, j = get_im2col_indices(images_shape, filter_shape, (pad_h, pad_w), stride)

    cols = cols.reshape(channels * np.prod(filter_shape), -1, batch_size)
    cols = cols.transpose(2, 0, 1)
    # Add column content to the images at the indices
    np.add.at(images_padded, (slice(None), k, i, j), cols)

    # Return image without padding
    return images_padded[:, :, pad_h[0]:height+pad_h[0], pad_w[0]:width+pad_w[0]]

class Dropout(Layer):
    def __init__(self, p = 0.3):
        self.p = p
        self._mark = None
        self.input_shape = None
        
    def forward_pass(self, X, training = True):
        # print("forward: Dropout",  X.shape)
        c = 1 - self.p
        if training:
            self._mark = np.random.uniform(size = X.shape) > self.p 
            c = self._mark
        return X * c

    def backward_pass(self, accum_grad):
        # print("backward: dropout", self._mark.shape, accum_grad.shape)
        return accum_grad * self._mark

    def output_shape(self):
        return self.input_shape
    

class Dense(Layer):
    def __init__(self, n_units, input_shape = None, learning_rate = 0.01):
        self.n_units = n_units
        self.input_shape = input_shape
        self.layer_input = None 
        self.W = None 
        self.bias = None 
        self.trainable = True
        self.learning_rate = learning_rate

    def initialize(self):
        self.W = np.random.uniform(size = (self.input_shape[0], self.n_units))
        self.bias = np.zeros((1,self.n_units))

    def forward_pass(self, X, training = True):
        self.layer_input = X 
        # print("forward: dense", X.shape)
        return np.dot(self.layer_input, self.W) + self.bias

    def backward_pass(self, accum_grad):
        W = self.W 
        if self.trainable:
            # print("backward: dense", self.layer_input.shape, accum_grad.shape)
            grad_W = np.dot(self.layer_input.T, accum_grad)
            grad_bias = np.sum(accum_grad, axis= 0, keepdims= True)

            self.W -= grad_W * self.learning_rate
            self.bias -= grad_bias * self.learning_rate
        return np.dot(accum_grad, W.T)

    def output_shape(self):
        return (self.n_units,)

class Flatten(Layer): #TODO: check input_shape vs pred_shape
    def  __init__(self, input_shape = None):
        self.input_shape = input_shape

    def forward_pass(self, X, training= True):
        self.input_shape = X.shape 
        return X.reshape((X.shape[0], -1))

    def backward_pass(self, accum_grad):
        return accum_grad.reshape(self.input_shape)

    def output_shape(self):
        return (np.prod(self.input_shape),)
