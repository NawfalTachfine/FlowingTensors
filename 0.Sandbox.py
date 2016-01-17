# -*- coding: utf-8 -*-

# First, let's make sure this thing is actually working
import tensorflow as tf 
yo = tf.constant('Whaazzzaaaaaaaaaa!!!')
sess = tf.Session()
print(sess.run(yo))

# Now for some brainy math
a = tf.constant(5)
b = tf.constant(6)
print( sess.run(a+b) )

# ----------------------------------------------------------------------------------------

# Let's try the provided dummy program. Maybe we'll figure out how this thing works then!

# classic
import numpy as np 

# setup the regression ingredients
x_data = np.random.rand(100).astype("float32")
y_data = x_data * 0.1 + 0.3

a = tf.Variable( tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable( tf.zeros([1]) )
y = a*x_data+b 

# optimization parameters
loss = tf.reduce_mean( tf.square( y-y_data ) )
optimizer = tf.train.GradientDescentOptimizer(0.5) # step size
train = optimizer.minimize(loss)

# doc says:
# "TensorFlow does not actually run any computation until the session is created and the run function is called."
# interesting..

init = tf.initialize_all_variables()

# lunching the graph, whatever that is
sess = tf.Session()
sess.run(init)

# time to fit that regression line
for step in xrange(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(a), sess.run(b))

# this actually doesn't look too bad, let's get real now!











