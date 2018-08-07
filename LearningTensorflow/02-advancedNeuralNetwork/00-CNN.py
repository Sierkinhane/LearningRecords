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
#image = tf.reshape(tf_x, [-1, 28, 28, 1])
tf_y = tf.placeholder(tf.int32, shape=[None, 10])

# create dataloader
dataset = tf.data.Dataset.from_tensor_slices((tf_x, tf_y))
dataset = dataset.shuffle(buffer_size=57000)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.repeat(3)
iterator = dataset.make_initializable_iterator()

images_labels = iterator.get_next()
b_x = tf.reshape(images_labels[0], [-1, 28, 28, 1])
# build cnn
conv1 = tf.layers.conv2d(
	inputs=b_x,
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

loss = tf.losses.softmax_cross_entropy(onehot_labels=images_labels[1], logits=output)
train_op = tf.train.AdamOptimizer(learning_rate=LR).minimize(loss)

accuracy = tf.metrics.accuracy(labels=tf.argmax(images_labels[1], axis=1), predictions=tf.argmax(output, axis=1))[1]

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer()) # the local var is for accuracy_op
sess.run(iterator.initializer, feed_dict={tf_x:mnist.train.images[:], tf_y:mnist.train.labels[:]})

accuracy_his = [[]]
saver = tf.train.Saver()

for step in range(2001):
	# b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
	try:
		'''
		train
		'''
		_, l = sess.run([train_op, loss])

		'''
		pick dataset's image and labels 
		'''
		# a, b = sess.run(images_labels)
		# plt.imshow(a[0].reshape(28,28))
		# print(b[0])
		# plt.show()

		if step % 50 == 0:
			'''
			validate
			'''
			accuracy_, l = sess.run([accuracy, loss], feed_dict={tf_x:test_x, tf_y:test_y})
			accuracy_his[0].append(accuracy_)
			print('step:[{0}/2000], the accuracy is {1:.4f}, loss is {2:.4f}'.format(step, accuracy_, l))

		if step % 2000 == 0:
			LR = 1/(1+(step/2000))*LR

		# if step % 1000 == 0:
		# 	saver.save(sess, './model/mnist-handwritten', global_step=step, write_meta_graph=False)
			
	except tf.errors.OutOfRangeError:
		print('3 epoch finished!')
		break

plt.plot(accuracy_his[0], label='accuracy')
plt.legend(loc='best')
'''
test
'''
test_output, images_labels = sess.run([output, images_labels], feed_dict={tf_x:test_x[:16], tf_y:test_y[:16]})
predictions = np.argmax(test_output, axis=1)
print(predictions, 'prediction number')
print(np.argmax(images_labels[1], axis=1), 'real number')

plt.show()

