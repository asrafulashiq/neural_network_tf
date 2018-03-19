import numpy as np

import tensorflow as tf

x = tf.placeholder(tf.float32, [2,])

w = tf.Variable(tf.zeros(2))

y = x + 2 - 2*w

out = {}
out["x"] = x
out["y"] = y

op = w.assign(w+y)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        w_v, o = sess.run([op, out], feed_dict={x:np.array([1, 0])})
        print(w_v)
        print(o)

