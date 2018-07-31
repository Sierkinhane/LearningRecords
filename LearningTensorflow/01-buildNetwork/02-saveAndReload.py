import tensorflow as tf 
import matplotlib.pyplot as plt 
import numpy as np 

tf.set_random_seed(1)
np.random.seed(1)

# fake data
x = np.linspace(-2, 2, 200)[:, np.newaxis]
noise = np.random.normal(0, 0.1, size=x.shape)
y = np.power(x, 2) + noise

def save():
	# data placeholder
	tf_x = tf.placeholder(tf.float32, x.shape)
	tf_y = tf.placeholder(tf.float32, y.shape)
	# build neural network
	hidden = tf.layers.dense(tf_x, 10, activation=tf.nn.relu)
	output = tf.layers.dense(hidden, 1)

	loss = tf.losses.mean_squared_error(labels=tf_y, predictions=output)
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
	train_op = optimizer.minimize(loss)

	with tf.Session() as sess:
		# init variables
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver()

		for step in range(400):
			sess.run(train_op, feed_dict={tf_x:x, tf_y:y})

		saver.save(sess, './model', write_meta_graph=False) # meta_graph is not recommeded

		# plotting
		pred, l = sess.run([output, loss], feed_dict={tf_x:x, tf_y:y})
		plt.figure(1, figsize=(10, 5))
		plt.subplot(121)
		plt.scatter(x, y)
		plt.plot(x, pred, 'r-', lw=5)
		plt.text(-0.8, 1.2, 'save loss=%.2f' %l, fontdict={'size':10, 'color':'red'})
		# plt.show()

def reload():
	
	# build entire net again and restore
	tf_x = tf.placeholder(tf.float32, x.shape)
	tf_y = tf.placeholder(tf.float32, y.shape)

	hidden = tf.layers.dense(tf_x, 10, activation=tf.nn.relu)
	output = tf.layers.dense(hidden, 1)

	loss = tf.losses.mean_squared_error(labels=tf_y, predictions=output)

	sess = tf.Session()
	saver = tf.train.Saver()
	saver.restore(sess, './model')

	# plotting
	pred, l = sess.run([output, loss], feed_dict={tf_x:x, tf_y:y})
	plt.subplot(122)	
	plt.scatter(x, y)
	plt.plot(x, pred, 'r-', lw=5)
	plt.text(-0.8, 1.2, 'reload loss=%.2f' %l, fontdict={'size':10, 'color':'red'})
	plt.show()

save()
tf.reset_default_graph()
reload()