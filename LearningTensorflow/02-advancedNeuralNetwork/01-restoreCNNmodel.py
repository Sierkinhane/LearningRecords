import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
#下载时出错
from tensorflow.examples.tutorials.mnist import input_data 

tf.set_random_seed(1)
np.random.seed(1)

BATCH_SIZE = 50
LR = 0.001

mnist = input_data.read_data_sets('./mnist', one_hot=True)
test_x = mnist.test.images[:2000]
test_y = mnist.test.labels[:2000]

# show an image
print(mnist.train.images.shape) # 55000, 784
print(mnist.train.labels.shape) # 55000, 10
# plt.imshow(mnist.train.images[2].reshape((28, 28)))
# plt.title('%i' % np.argmax(mnist.train.labels[2]))

tf_x = tf.placeholder(tf.float32, shape=[None, 28*28])
image = tf.reshape(tf_x, [-1, 28, 28, 1])
tf_y = tf.placeholder(tf.int32, shape=[None, 10])

# build cnn
conv1 = tf.layers.conv2d(
	inputs=image,
	filters=16,
	kernel_size=5,
	strides=1,
	padding='same',
	activation=tf.nn.relu,
	kernel_regularizer=tf.nn.l2_normalize
	) # 28*28*16

pool1 = tf.layers.max_pooling2d(
	inputs=conv1,
	pool_size=2,
	strides=2
	) # 14*14*16

	  # 14*14*32
conv2 = tf.layers.conv2d(pool1, 32, 5, 1, 'same', activation=tf.nn.relu, kernel_regularizer=tf.nn.l2_normalize)
pool1 = tf.layers.max_pooling2d(conv2, 2, 2) # 7*7*32
flat = tf.reshape(pool1, [-1, 7*7*32])
output = tf.layers.dense(flat, 10, kernel_regularizer=tf.nn.l2_normalize) # 用softmax函数激活导致loss=1.5.. 不能下降的原因？

loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)
train_op = tf.train.AdamOptimizer(learning_rate=LR).minimize(loss)

accuracy = tf.metrics.accuracy(labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1))[1]

# restore model also initialize the variables
sess = tf.Session()
# sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer()) # the local var is for accuracy_op
										   # local variables stored in the disk
# restore the model
saver = tf.train.Saver()
saver.restore(sess, './model/mnist-handwritten-2000')

accuracy_, l = sess.run([accuracy, loss], feed_dict={tf_x:test_x, tf_y:test_y})
print(accuracy_, l)

'''
test
'''
test_output = sess.run(output, feed_dict={tf_x:test_x[:16], tf_y:test_y[:16]})
predictions = np.argmax(test_output, axis=1)
print(predictions, 'prediction number')
print(np.argmax(test_y[:16], axis=1), 'real number')

plt.show()

