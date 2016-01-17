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
# shape is optional but VERY handy
x_ = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

# Initialize all variables
sess.run(tf.initialize_all_variables())