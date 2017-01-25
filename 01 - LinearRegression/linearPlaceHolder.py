import tensorflow as tf

# data set
x_data = [1., 2., 3., 4.]
y_data = [2., 4., 6., 8.]

# try to find values for w and b that compute y_data = W * x_data + b
# range is -100 ~ 100
W = tf.Variable(tf.random_uniform([1], -10000., 10000.))
b = tf.Variable(tf.random_uniform([1], -10000., 10000.))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
###############################
# tf.placeholder(dtype, shape=None, name=None)
###############################
# Inserts a placeholder for a tensor that will be always fed.
#
# Important: This tensor will produce an error if evaluated. Its value must be fed using the feed_dict optional argument to Session.run(), Tensor.eval(), or Operation.run().
#
# For example:
#
#   x = tf.placeholder(tf.float32, shape=(1024, 1024))
#   y = tf.matmul(x, x)
#
#   with tf.Session() as sess:
#   print(sess.run(y))  # ERROR: will fail because x was not fed.
#
#   rand_array = np.random.rand(1024, 1024)
#   print(sess.run(y, feed_dict={x: rand_array}))  # Will succeed.
#
# Args:
#
#   dtype: The type of elements in the tensor to be fed.
#   shape: The shape of the tensor to be fed (optional). If the shape is not specified, you can feed a tensor of any shape.
#   name: A name for the operation (optional).
#
# Returns:
#
#   A Tensor that may be used as a handle for feeding a value, but not evaluated directly.
###############################


# my hypothesis
hypothesis = W * X + b

# Simplified cost function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# minimize
a = tf.Variable(0.1)  # learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)  # goal is minimize cost

# before starting, initialize the variables
init = tf.global_variables_initializer()

# launch
sess = tf.Session()
sess.run(init)

# fit the line
for step in range(2001):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 20 == 0:
        print( step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W), sess.run(b) )

print( sess.run(hypothesis, feed_dict={X: 5}) )
print( sess.run(hypothesis, feed_dict={X: 2.5}) )
