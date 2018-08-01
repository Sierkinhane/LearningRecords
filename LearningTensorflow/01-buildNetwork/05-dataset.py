import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt

x = np.random.uniform(-1, 1, (1000, 1))
y = np.power(x, 2) + np.random.normal(0, 0.1, size=x.shape)
x_train, x_val = np.split(x, [800])
y_train, y_val = np.split(y, [800])

tf_x = tf.placeholder(tf.float32, shape=[None, 1])
tf_y = tf.placeholder(tf.float32, shape=[None, 1])

# create dataloader
dataset = tf.data.Dataset.from_tensor_slices((tf_x, tf_y))
dataset = dataset.shuffle(buffer_size=1000)
dataset = dataset.batch(32) # 800/32 剩下的舍去
dataset = dataset.repeat(4) # repeat 4 epoch
iterator = dataset.make_initializable_iterator()

# build neural network
b_x, b_y = iterator.get_next()
hidden = tf.layers.dense(b_x, 20, activation=tf.nn.relu, name='hidden')
output = tf.layers.dense(hidden, 1, name='output')
loss = tf.losses.mean_squared_error(b_y, output)
optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9)
train_op = optimizer.minimize(loss)


sess = tf.Session()
sess.run([iterator.initializer, tf.global_variables_initializer()],  feed_dict={tf_x:x_train, tf_y:y_train})

loss_his = [[]]
for step in range(200):
	try:
		_, train_loss = sess.run([train_op, loss])
		if step % 10 == 0:
			val_loss = sess.run(loss, feed_dict={tf_x:x_val, tf_y:y_val})
			loss_his[0].append(val_loss)
			print('step:{0}/200, train loss:{1:.2f}, validate loss:{2:.2f}'.format(step, train_loss, val_loss))
	except tf.errors.OutOfRangeError:
		print('finished')
		break
plt.plot(loss_his[0], label='val_loss')
plt.legend(loc='best')
plt.show()