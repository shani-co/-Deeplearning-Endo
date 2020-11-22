
import tensorflow._api.v2.compat.v1 as tf

tf.disable_v2_behavior()


@tf.custom_gradient

def loss_layer(var_w, const_x, const_y):
  def grad(delta):
     #partial derivative according to var_w, and according to const_x, const_y (which aren't in use)
     return tf.reduce_mean(2*((var_w * const_x) - const_y) * const_x * delta), None, None
  return tf.reduce_mean(tf.pow(tf.multiply(var_w, const_x)- const_y, 2)), grad
  loss = tf.reduce_mean(tf.pow(y - y_, 2))
x = tf.constant([[7.01], [3.02], [4.99], [8.]])
y_ = tf.constant([[14.01], [6.01], [10.], [16.04]])
w = tf.Variable(0.1)  # note the dot
loss = loss_layer(w, x, y_)
grad_loss = tf.gradients(loss, w)
update = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(0, 400):
  print(sess.run([w,grad_loss, loss, update]))
print(sess.run(w))
