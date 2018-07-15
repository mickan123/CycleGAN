import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import random
from ops import *
from model import *
from arguments import parse_arguments

class model(object):

	def __init__(self, args):
		self.image_dim = int(args.id) #output image size is IMAGE_DIM x IMAGE_DIM
		self.images_A = load_images(self.image_dim, args.da) #Image dataset
		self.images_B = load_images(self.image_dim, args.db) #Image dataset

		self.disc_iterations = int(args.di) #Number of iterations to train disc for per gen iteration
		self.max_iterations = int(args.i) #Max iterations to train for
		self.save_interval = int(args.s) #Save model every save_interval epochs
		self.print_interval = int(args.p) #How often we print progress
		
		self.mb_size = int(args.mb) #Minibatch size
		self.Z_dim = int(args.z) #Noise vector dimensions
		self.mult = float(args.m) #Scalar multiplier for model size
		self.loss = args.l #Loss function to use
		self.decay_lr = args.dlr #Boolean on whether to use decaying learning rate
		
		self.n = 0 #Minibatch seed
		self.it = 0 #Current iteration
		self.learning_rate = 2e-4

		self.load_model = args.lm #Model to load

		#create output directory
		if not os.path.exists('out/'):
			os.makedirs('out/')

	def __call__(self):
		self.build_model()
		self.train_model()

	def build_model(self):

		with tf.variable_scope("Model") as scope:

			self.input_A = tf.placeholder(tf.float32, [None, self.image_dim, self.image_dim, 3], name="inputA")
			self.input_B = tf.placeholder(tf.float32, [None, self.image_dim, self.image_dim, 3], name="inputB")

			self.gen_B = build_generator(self.input_A, name = "GenAtoB")
			self.gen_A = build_generator(self.input_B, name = "GenBtoA")
			self.dec_A = build_discriminator(self.input_A, name = "DiscA")
			self.dec_B = build_discriminator(self.input_B, name = "DiscB")

			scope.reuse_variables()

			self.dec_gen_A = build_discriminator(self.gen_A, "DiscA")
			self.dec_gen_B = build_discriminator(self.gen_B, "DiscB")
			self.cyc_A = build_generator(self.gen_B, "GenBtoA")
			self.cyc_B = build_generator(self.gen_A, "GenAtoB")

			scope.reuse_variables()

		with tf.variable_scope("Loss") as scope:
			self.build_optimizer()


	def build_optimizer(self):

		#Should predict 1 for true samples to discriminator A and B
		D_A_loss_1 = tf.reduce_mean(tf.squared_difference(self.dec_A, 1))
		D_B_loss_1 = tf.reduce_mean(tf.squared_difference(self.dec_B, 1))

		#Should predict 0 for generated images 
		D_A_loss_2 = tf.reduce_mean(tf.square(self.dec_gen_A))
		D_B_loss_2 = tf.reduce_mean(tf.square(self.dec_gen_B))

		#Discriminator wants to minimize both of the above losses
		d_A_loss = D_A_loss_1 + D_A_loss_2
		d_B_loss = D_B_loss_1 + D_B_loss_2 

		#Generator wants discriminator to predict 1 for generated images
		g_loss_B_1 = tf.reduce_mean(tf.squared_difference(self.dec_gen_A, 1))
		g_loss_A_1 = tf.reduce_mean(tf.squared_difference(self.dec_gen_B, 1))

		#Cyclic loss wants to reduce difference between input and input after cycle of GAN
		cyc_loss = tf.reduce_mean(tf.abs(self.input_A - self.cyc_A)) + tf.reduce_mean(tf.abs(self.input_B - self.cyc_B))

		#Generator wants to minimize generator loss and cyclic loss
		g_loss_A = g_loss_A_1 + 10 * cyc_loss
		g_loss_B = g_loss_B_1 + 10 * cyc_loss

		optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1 = 0.5)

		self.model_vars = tf.trainable_variables()

		d_A_vars = [var for var in self.model_vars if 'DiscA' in var.name]
		g_A_vars = [var for var in self.model_vars if 'GenA' in var.name]
		d_B_vars = [var for var in self.model_vars if 'DiscB' in var.name]
		g_B_vars = [var for var in self.model_vars if 'GenB' in var.name]

		self.d_A_trainer = optimizer.minimize(d_A_loss, var_list = d_A_vars)
		self.d_B_trainer = optimizer.minimize(d_B_loss, var_list = d_B_vars)
		self.g_A_trainer = optimizer.minimize(g_loss_A, var_list = g_A_vars)
		self.g_B_trainer = optimizer.minimize(g_loss_B, var_list = g_B_vars)

	#Returns next batch of images
	def next_batch(self, images, seed):
		index = (seed) % (len(images) // self.mb_size)
		mb = images[index * self.mb_size:(index + 1) * self.mb_size]
		return mb

	#Generate and save some samples to out/ folder and print iteration + time taken
	def generate_statistics(self):
		batch = self.next_batch(self.images_A, random.randint(1,10000))[:8]
		generated = self.sess.run(self.gen_B, feed_dict = {self.input_A : batch})
		images = [val for pair in zip(batch, generated) for val in pair]
		fig = plot(images, self.image_dim)
		plt.savefig('out/{}.png'.format(str(self.it / self.print_interval).zfill(3)), bbox_inches = 'tight')
		plt.close(fig)

		print('Iter: {}'.format(self.it))
		self.end_time = time.time()
		print("Time taken: " + str(self.end_time - self.start_time))
		self.start_time = self.end_time
		print()

	#Trains the model
	def train_model(self):
		self.start_time = time.time()

		with tf.Session() as self.sess:

			self.sess.run(tf.global_variables_initializer())

			self.saver = tf.train.Saver()
			
			start_it = 0
			if (self.load_model != None):
				self.saver = tf.train.import_meta_graph('Training Model/' + self.load_model)
				self.saver.restore(self.sess, tf.train.latest_checkpoint('Training Model/'))
				#Extract the iteration from load_model string
				start_it = int((self.load_model.split('.')[-2]).split('-')[-1])

			for self.it in range(start_it, self.max_iterations):

				feed = {self.input_A : self.next_batch(self.images_A, self.it), 
                        self.input_B : self.next_batch(self.images_B, self.it)}

				#Train generator G_A->B
				_, gen_B_temp = self.sess.run([self.g_A_trainer, self.gen_B], feed_dict = feed)
                                               
				#Train discriminator B
				_ = self.sess.run([self.d_B_trainer], feed_dict = feed)
                                   
				#Train generator G_B->A
				_, gen_A_temp = self.sess.run([self.g_B_trainer, self.gen_A], feed_dict = feed)
                                               
				#Train discriminator A
				_ = self.sess.run([self.d_A_trainer], feed_dict = feed)
                                   
				if (self.it % self.print_interval == 0):
					self.generate_statistics()
					
				if (self.it % self.save_interval == 0):
					self.saver.save(self.sess, 'Training Model/train_model', global_step = self.it)

			self.saver.save(self.sess, 'Final Model/Final_model')
def main():
	args = parse_arguments()
	GAN = model(args)
	GAN()

if __name__ == '__main__':
	main()

