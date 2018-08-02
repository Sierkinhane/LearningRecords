import numpy as np 
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
import matplotlib.pyplot as plt 


np.random.seed(1)

# create data
x = np.linspace(-2, 2, 200)[:, np.newaxis]
#np.random.shuffle(x)
noise = np.random.normal(0, 0.3, x.shape)
y = np.power(x, 2) + noise

# show data distribution
# plt.scatter(x, y)
# plt.show()

# build a neural network from the 1st layes to the last layers
model = Sequential()
model.add(Dense(units=10, activation='relu', input_dim=1))  # input_dim 指的是x的维数
model.add(Dense(units=1))

# choose loss function and optimizing method
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='mse')

# training
plt.ion()
for step in range(1001):
	# cost = model.fit(x, y)
	cost = model.train_on_batch(x, y)
	# cost = model.fit(x, y)
	y_pred = model.predict(x)

	if step % 50 == 0:
		print('[{0}/1000], train loss {1}'.format(step, cost))
		plt.cla()
		plt.scatter(x, y, label='regression')
		plt.legend(loc='best')
		plt.plot(x, y_pred, 'r-', lw=5)
		plt.pause(0.1)
		

plt.ioff()
plt.show()