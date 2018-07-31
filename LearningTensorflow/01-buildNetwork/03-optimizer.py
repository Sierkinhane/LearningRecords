import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt
import time

# fake data
x = np.linspace(-2, 2, 200)[:, np.newaxis]
noise = np.random.normal(0, 0.5, x.shape)
y = np.power(x, 2) + noise

plt.scatter(x, y)
plt.show()

# network class
class  Net():
	def __init__(self, optimizer):
		self.x = tf.placeholder(tf.float32, [None, 1])
		self.y = tf.placeholder(tf.float32, [None, 1])
		hidden = tf.layers.dense(self.x, 20, activation=tf.nn.relu)
		output = tf.layers.dense(hidden, 1)
		self.loss = tf.losses.mean_squared_error(self.y, output)
		self.train = optimizer.minimize(self.loss)


# different optimizer
SGD = Net(tf.train.GradientDescentOptimizer(learning_rate=0.01))
Momentum = Net(tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9))
RMSprop = Net(tf.train.RMSPropOptimizer(learning_rate=0.01))
Adam = Net(tf.train.AdamOptimizer(learning_rate=0.01))
nets = [SGD, Momentum, RMSprop, Adam]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

losses_his = [[],[],[],[]]
total_time = [0, 0, 0, 0]

for step in range(500):
	index = np.random.randint(0, x.shape[0], 64)		
	b_x = x[index]
	b_y = x[index]

	i=0
	for net, loss_his in zip(nets, losses_his):
		start = time.time()
		_, loss = sess.run([net.train, net.loss], feed_dict={net.x:b_x, net.y:b_y})
		end = time.time()
		total_time[i]+=(end-start)
		loss_his.append(loss)
		i+=1

print(total_time)
# plot loss history
labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
for i, loss_his in enumerate(losses_his):
	plt.plot(loss_his, label=labels[i])
plt.legend(loc='best')
plt.ylim((0, 0.5))
plt.show()
