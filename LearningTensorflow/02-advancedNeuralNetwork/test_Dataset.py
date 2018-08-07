import numpy as np 
import tensorflow as tf 

x = np.array([[1, 2, 3],
			  [2, 3, 4],
			  [4, 5, 6],
			  [6, 7, 7]])
y = np.array([[0, 0, 1],
			  [1, 0, 0],
			  [0, 1, 0],
			  [1, 1, 0]])
tf_x = tf.placeholder(tf.float32, shape=[4, 3])
tf_y = tf.placeholder(tf.int32, shape=[4, 3])
dataset = tf.data.Dataset.from_tensor_slices((tf_x, tf_y))
dataset = dataset.shuffle(2)
dataset = dataset.batch(2)
# dataset = dataset.repeat(3)
iterator = dataset.make_initializable_iterator()
x_temp, y_temp = iterator.get_next()

sess = tf.Session()
sess.run(iterator.initializer, feed_dict={tf_x:x, tf_y:y})
for i in range(2):
	try:
		print(sess.run(x_temp))
		print(sess.run(y_temp))
		print(sess.run(x_temp)[0])
		print(sess.run(y_temp)[0])
		
	except tf.errors.OutOfRangeError:
		pass