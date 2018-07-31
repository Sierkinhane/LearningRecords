import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 

# fake data 
x = np.linspace(-5, 5, 200)  # shape=(100, 1)

# following are popular activation functions
y_relu = tf.nn.relu(x)
y_sigmoid = tf.nn.sigmoid(x)
y_tanh = tf.nn.tanh(x)
y_softplus = tf.nn.softplus(x)

sess = tf.Session()
y_relu, y_sigmoid, y_tanh, y_softplus = sess.run([y_relu, y_sigmoid, y_tanh, y_softplus])

# plt to visualize these activation function
plt.figure(1, figsize=(8, 6))
plt.subplot(221)
plt.plot(x, y_relu, 'red', label='relu')
plt.ylim((-1, 5))
plt.legend(loc='best')

plt.subplot(222)
plt.plot(x, y_sigmoid, 'blue', label='sigmoid')
plt.ylim((-0.2, 1))
plt.legend(loc='best')

plt.subplot(223)
plt.plot(x, y_tanh, 'green', label='tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc='best')

plt.subplot(224)
plt.plot(x, y_softplus, 'black', label='softplus')
plt.legend(loc='best')

plt.show()