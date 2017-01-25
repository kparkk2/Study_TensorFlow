import tensorflow as tf

x_data = [1., 2., 3.]
y_data = [1., 2., 3.]

# try to find values for w and b that compute y_data = W * x_data + b
w = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

###############################
# tf.random_uniform(shape, minval=0, maxval=None, dtype=tf.float32, seed=None, name=None)
###############################
# Args
#
#   shape: A 1-D integer Tensor or Python array. The shape of the output tensor.
#   minval: A 0-D Tensor or Python value of type dtype. The lower bound on the range of random values to generate. Defaults to 0.
#   maxval: A 0-D Tensor or Python value of type dtype. The upper bound on the range of random values to generate. Defaults to 1 if dtype is floating point.
#   dtype: The type of the output: float32, float64, int32, or int64.
#   seed: A Python integer. Used to create a random seed for the distribution. See set_random_seed for behavior.
#   name: A name for the operation (optional).
###############################

# my hypothesis
hypothesis = w * x_data + b

# Simplified cost function
cost = tf.reduce_mean(tf.square(hypothesis - y_data))

# minimize
a = tf.Variable(0.1)  # learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# before starting, initialize the variables
init = tf.global_variables_initializer()

# launch
sess = tf.Session()
sess.run(init)

# fit the line
for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print( step, sess.run(cost), sess.run(w), sess.run(b) )

# learns best fit is w: [1] b: [0]
