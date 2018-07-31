import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
#下载时出错
from tensorflow.examples.tutorials.mnist import input_data 

tf.set_random_seed(1)
np.random.seed(1)

BATCH_SIZE = 16
LR = 0.001

mnist = input_data.read_data_sets('./mnist', one_hot=True)
test_x = mnist.test.images[:2000]
test_y = mnist.test.labels[:2000]

# show an image
print(mnist.train.images.shape) # 55000, 784
print(mnist.train.labels.shape) # 55000, 10
#plt.imshow(mnist.train.images[0].reshape((28, 28)))
#plt.title('%i' % np.argmax(mnist.train.labels[0]))

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
	activation=tf.nn.relu
	) # 28*28*16

pool1 = tf.layers.max_pooling2d(
	inputs=conv1,
	pool_size=2,
	strides=2
	) # 14*14*16

	  # 14*14*32
conv2 = tf.layers.conv2d(pool1, 32, 5, 1, 'same', activation=tf.nn.relu)
pool1 = tf.layers.max_pooling2d(conv2, 2, 2) # 7*7*32
flat = tf.reshape(pool1, [-1, 7*7*32])
output = tf.layers.dense(flat, 10, activation=tf.nn.softmax)

loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)
train_op = tf.train.AdamOptimizer(learning_rate=LR).minimize(loss)

accuracy = tf.metrics.accuracy(labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1))[1]

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer()) # the local var is for accuracy_op

accuracy_his = [[]]
for step in range(10001):
	b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
	_, l = sess.run([train_op, loss], feed_dict={tf_x:b_x, tf_y:b_y})
	if step % 50 == 0:
		accuracy_, l = sess.run([accuracy, loss], feed_dict={tf_x:test_x, tf_y:test_y})
		accuracy_his[0].append(accuracy_)
		print('step:[{0}/10000], the accuracy is {1:.4f}, loss is {2:.4f}'.format(step, accuracy_, l))
	if step % 2000 == 0:
		LR = 1/(1+(step/2000))*LR

plt.plot(accuracy_his[0], label='accuracy')
plt.legend(loc='best')
test_output = sess.run(output, feed_dict={tf_x:test_x[:10]})
predictions = np.argmax(test_output, axis=1)
print(predictions, 'prediction number')
print(np.argmax(test_y[:10], axis=1), 'real number')

plt.show()

