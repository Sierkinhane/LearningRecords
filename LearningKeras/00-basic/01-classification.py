import keras
from keras.models import Sequential 
from keras.layers import Dense, Activation
import keras.optimizers as optimizers
import numpy as np
import matplotlib.pyplot as plt 

# create dummy data
n_data = np.ones((100, 2))
x0 = np.random.normal(2*n_data, 1)
y0 = keras.utils.to_categorical(np.ones(100), num_classes=2)
x1 = np.random.normal(-2*n_data, 1)
y1 = keras.utils.to_categorical(np.zeros(100), num_classes=2)

# train data and labels
x = np.concatenate((x0, x1))
y = np.concatenate((y0, y1))

# plotting
plt.scatter(x[:,0], x[:,1], c=y[:,0], cmap='RdYlGn')
plt.show()

model = Sequential([
	Dense(10, input_dim=2),
	Activation('relu'),
	Dense(2),
	Activation('softmax')
])

# build learning process
sgd = optimizers.Adam(lr=0.001)
model.compile(
	optimizer=sgd, 
	loss='categorical_crossentropy',
	metrics=['accuracy']
)

plt.ion()
for step in range(1001):
	loss, accuracy = model.train_on_batch(x, y)
	pred = model.predict(x)

	if step % 10 == 0:
		plt.cla()
		plt.scatter(x[:,0], x[:,1], c=pred[:,0], cmap='RdYlGn', label='Classification')
		plt.legend(loc='best')
		print('[{0}/1000], loss: {1:.6f}, accuracy: {2:.4f}'.format(step,accuracy, loss))
		plt.pause(0.1)

plt.ioff()
plt.show()

from keras import backend 
backend.clear_session()
# 解决下面的异常
# Exception ignored in: <bound method BaseSession.__del__ of <tensorflow.python.client.session.Session object at 0x0000019E62AB65C0>>
# Traceback (most recent call last):
#   File "F:\Anaconda\anaconda3-5.2.0\lib\site-packages\tensorflow\python\client\session.py", line 702, in __del__
# TypeError: 'NoneType' object is not callable