import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob
from scipy import misc
from random import shuffle


#Loads images dataset normalizing and resizing them
def load_images(size, path):
	image_list = []
	for i, image_path in enumerate(glob.glob(path + "/*")):
		image = misc.imread(image_path)
		image = misc.imresize(image, [size, size])
		image = (image / 127.5) - 1
		image_list.append(image)
	
	shuffle(image_list)

	return image_list

#Leaky relu function
def lrelu(x, th=0.2):
    return tf.maximum(th * x, x)

#Returns 4x4 plot of image samples
def plot(samples, dim):
	fig = plt.figure(figsize=(4, 4))
	gs = gridspec.GridSpec(4, 4)
	gs.update(wspace=0.05, hspace=0.05)
	for i, sample in enumerate(samples):
		sample = (sample + 1) / 2
		ax = plt.subplot(gs[i])
		plt.axis('off')
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.set_aspect('equal')
		plt.imshow(sample.reshape(dim, dim, 3), cmap='Greys_r')

	return fig