# -*- coding: utf-8 -*-

# Last edit: 17/01/2015

# Source: https://www.tensorflow.org/versions/master/tutorials/mnist/pros/index.html


# Skipping the noob version because I like to live dangerously

# Get them data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Appearently we're going for an interactive session this time
import tensorflow as tf
sess = tf.InteractiveSession()

# ----------------------------------------------------------------------------------------

# Part one: Softmax Regression Model / Single Linear Layer

# Placeholders specify values that are entered upon running the computations
# shape is optional but VERY handy for debugging
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# Initialize all variables
sess.run(tf.initialize_all_variables())

# Predicted Class and Cost Function
y = tf.nn.softmax( tf.matmul(x,W) + b )
cross_entropy = - tf.reduce_sum( y_*tf.log(y) )
# daaang is this concise!
# tf.reduce_sum sums across all images in the minibatch, as well as all classes

# Training the Model
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# (new computation added to graph)

for i in range(1000):
	batch = mnist.train.next_batch(50)
	train_step.run( feed_dict={x: batch[0], y_:batch[1]} )
	print 'This gradient is going down, down, down! Iteration:', i, ' '
# feed_dict replaces any tensor in computation graph, particularly placeholders

# Evaluating the Model
correct_prediction = tf.equal( tf.argmax(y,1), tf.argmax(y_,1) )
accuracy = tf.reduce_mean( tf.cast(correct_prediction, "float") )
print 'How accurate is the model? Let\'s ask the test set...'
print( accuracy.eval( feed_dict={x:mnist.test.images, y_:mnist.test.labels} ) )
print 'Meh, we can do better than that!'


# ----------------------------------------------------------------------------------------

# Part two: Multilayer Convolutional Network

# Sh*t's getting serious!

# Weight Initialization
# use small amounts of noize for symmetry breaking and gradients fading to 0
# slightly positive biases help avoid dead neurons

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

# Convolution and Pooling

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# using a stride of 1 and a padding of 0 for the convolutions, and pooling over 2x2 blocks

# First Convolutional Layer
# we're computing 32 features for each 5x5 patch, with 1 input channel and 32 output channels
# the bias vector has a value for each output channel
W_conv1 = weight_variable([5,5,1,32]) 
b_conv1 = bias_variable([32]) 

# convolve, add bias, apply ReLU and max pool
x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu( conv2d(x_image, W_conv1) + b_conv1 )
h_pool1 = max_pool_2x2(h_conv1)

# Second Convolutional Layer
# now we're having 64 features for each 5x5 patch
W_conv2 = weight_variable([5,5,32,64]) 
b_conv2 = bias_variable([64]) 

h_conv2 = tf.nn.relu( conv2d(h_pool1, W_conv2) + b_conv2 )
h_pool2 = max_pool_2x2(h_conv2)

# Densely Connected Layer
# image size at this point: 7x7
# last layer: fullty connected with 1024 neurons => process entire image

W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu( tf.matmul(h_pool2_flat, W_fc1) + b_fc1 )

# Dropout
# applied before the readout layer to reduce overfitting
# the placeholder for the probability that a neuron is kept is on/training and off/testing
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout Layer (softmax)
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# Training and Evaluation

cross_entropy = - tf.reduce_sum( y_*tf.log(y_conv) )
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal( tf.argmax(y_conv,1), tf.argmax(y_,1) )
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())

for i in range(20000):
	batch = mnist.train.next_batch(50)
	if i%100 == 0:
		train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})
		print("step %d, training accuracy %g"%(i, train_accuracy))
	train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0}))








