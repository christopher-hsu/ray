import tensorflow as tf
import numpy as np
import pdb

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial, name="W")

def bias_variable(shape, init_values=None):
	if init_values is not None:
		initial = tf.constant(init_values, dtype = tf.float32)
	else:
		initial = tf.constant(0.1, shape=shape)
 	return tf.Variable(initial, name="b")


class MLP():
	def __init__(
		self,
		name,
		input_dim, 
		output_dim,
		num_layers,
		hidden_size,
		input_given = None,
		hidden_nonlinearity=tf.nn.relu,
		output_nonlinearity=tf.identity,
		output_bias = None,
		):
		assert(len(input_dim)==1) # For now..
		assert(len(output_dim)==1)
		with tf.variable_scope(name):
			if input_given is not None:
				input_layer = input_given
			else:
				input_layer = tf.placeholder(tf.float32, shape=[None,input_dim[0]], name = "input")
			prev_shape = input_dim[0]
			layers = [input_layer]
			for i in range(num_layers):
				with tf.name_scope("hidden_%d"%i):	
					layer = hidden_nonlinearity(tf.matmul(layers[i], 
								weight_variable((prev_shape, hidden_size)))+ bias_variable((hidden_size,)))
				layers.append(layer)
				prev_shape = hidden_size
			#self.keep_prob = tf.placeholder(tf.float32)
			#layer = tf.nn.dropout(layers[-1], self.keep_prob)
			#layers.append(layer)
			with tf.name_scope("output"):
				output_layer = output_nonlinearity(tf.matmul(layers[-1], 
									weight_variable((hidden_size,output_dim[0]))) \
									+ bias_variable(output_dim, output_bias))
			self.layers = layers[1:]
			self.l_input = input_layer
			self.l_output = output_layer

	def get_input_layer(self):
		return self.l_input

	def get_output(self):
		return self.l_output

class ConvNet():
	def __init__(
		self,
		name,
		input_dim, 
		output_dim, 
		hidden_sizes, # Fully connected layers - # of units in each layer; (1st, 2nd, ...)
		conv_filters,
		patch_sizes, #  Assuming that filter_height = filter_width
		conv_strides, 
		conv_padding = "VALID",
		pool_strides=(2,2),		
		ksizes=(2,2),
		pool_padding = None,
		input_given = None,
		hidden_nonlinearity=tf.nn.relu,
		output_nonlinearity=tf.identity,
		output_bias = None
		):
		n_filters = len(conv_filters)
		conv_paddings = (conv_padding,)*n_filters
		pool_paddings = (pool_padding,)*n_filters
		assert(len(input_dim)==3)
		with tf.variable_scope(name):
			if input_given is not None:
				input_layer = input_given
			else:
				input_layer = tf.placeholder(tf.float32, shape=[None]+list(input_dim), name = "input")

			layers = [input_layer]
			input_shape = np.array(input_dim, dtype=np.int32)
			for i in range(n_filters):
				with tf.name_scope("conv_%d"%i):
					conv = tf.nn.conv2d(layers[-1], 
										weight_variable([patch_sizes[i], patch_sizes[i], input_shape[-1], conv_filters[i]]),
										strides=[1, conv_strides[i], conv_strides[i], 1], 
										padding=conv_paddings[i])
					h_conv = hidden_nonlinearity(conv + bias_variable([conv_filters[i]]))
					#input_shape = (input_shape[:-1]-patch_sizes[i]+2*int(conv_paddings[i]=="ZERO"))/conv_strides[i]+1
					output_h, output_w = conv_output_size(input_shape[:2], [conv_strides[i]]*2, [patch_sizes[i]]*2, conv_paddings[i])
					if pool_paddings[i] is not None:
						layer = tf.nn.max_pool(h_conv, 
												ksize=[1, ksizes[i], ksizes[i],1],
												strides=[1, pool_strides[i], pool_strides[i],1],
												padding=pool_paddings[i])
						output_h, output_w = conv_output_size([output_h, output_w], [pool_strides[i]]*2, [ksizes[i]]*2, pool_paddings[i])
						layers.append(layer)
					else: # No Pooling
						layers.append(h_conv)
					input_shape = np.array([output_h, output_w, conv_filters[i]], dtype=np.int32)
					
			input_shape = np.prod(input_shape)
			layers.append(tf.reshape(layers[-1], [-1, input_shape]))

			for i in range(len(hidden_sizes)):
				with tf.name_scope("hidden_%d"%i):
					W = weight_variable([input_shape, hidden_sizes[i]])
					b = bias_variable([hidden_sizes[i]])
					layers.append(hidden_nonlinearity(tf.matmul(layers[-1], W)+b))
					input_shape = hidden_sizes[i]

			# Dropout
			#self.keep_prob = tf.placeholder(tf.float32)
			#layers.append(tf.nn.dropout(layers[-1], self.keep_prob))
			with tf.name_scope("output"):
				output_layer = output_nonlinearity(tf.matmul(layers[-1], 
									weight_variable((input_shape ,output_dim[0]))) \
									+ bias_variable(output_dim, output_bias))
			self.all_layers = layers
			self.l_input = input_layer
			self.l_output = output_layer

	def get_input_layer(self):
		return self.l_input

	def get_output(self):
		return self.l_output

def conv_output_size(input2Dshape, stride2Dshape, patch_size, padding):
	# input2Dshape = [input_height, input_width]
	# stride2Dshape = [stride[1], stride[2]]
	# patch_size = [patch_height, patch_width]
	# padding = "VALID" or "SAME"
	bool_padding = int(padding == "VALID")
	out_height = np.ceil(float(input2Dshape[0] - bool_padding*(patch_size[0]-1)) / float(stride2Dshape[0]))
	out_width  = np.ceil(float(input2Dshape[1] - bool_padding*(patch_size[1]-1)) / float(stride2Dshape[1]))

	return out_height, out_width





