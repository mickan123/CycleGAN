import tensorflow as tf
from ops import lrelu

def resnet_block(input_res, dim):
	out_res = tf.layers.conv2d(input_res, dim, 3, 1, padding = 'same', activation = lrelu)
	out_res = tf.layers.conv2d(input_res, dim, 3, 1, padding = 'same', activation = lrelu)
	return input_res + out_res

#Builds a generator model for our cycleGAN
def build_generator(inputs, name):

	with tf.variable_scope(name):
		conv_1 = tf.layers.conv2d(inputs, 32, 3, 1, padding = 'same', activation = lrelu)
		conv_2 = tf.layers.conv2d(conv_1, 32, 3, 2, padding = 'same', activation = lrelu)
		conv_3 = tf.layers.conv2d(conv_2, 32, 3, 2, padding = 'same', activation = lrelu)
		net = conv_3
		for _ in range(4):
			net = resnet_block(net, 32)
		net += conv_3

		conv_4 = tf.layers.conv2d_transpose(net, 32, 3, 2, padding = 'same', activation = lrelu)
		conv_5 = tf.layers.conv2d_transpose(conv_4, 32, 3, 2, padding = 'same', activation = lrelu)
		conv_6 = tf.layers.conv2d_transpose(conv_5, 3, 3, 1, padding = 'same', activation = lrelu)
		out = tf.nn.tanh(conv_6)

		return out

#Builds a discriminator model for our cycleGAN
def build_discriminator(inputs, name):

	with tf.variable_scope(name):
		conv_1 = tf.layers.conv2d(inputs, 32, 4, 2, padding = 'same', activation = lrelu)
		conv_2 = tf.layers.conv2d(conv_1, 64, 4, 2, padding = 'same', activation = lrelu)
		conv_3 = tf.layers.conv2d(conv_2, 128, 4, 2, padding = 'same', activation = lrelu)
		conv_4 = tf.layers.conv2d(conv_3, 256, 4, 2, padding = 'same', activation = lrelu)
		
		conv_5 = tf.layers.conv2d(conv_4, 1, 4, 1, padding = 'same', activation = lrelu)

		flat = tf.layers.Flatten()(conv_5)
		logits = tf.layers.dense(flat, 1)

		return logits