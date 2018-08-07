import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
import keras.optimizers as optimizers
import numpy as np
import matplotlib.pyplot as plt
import time

BATCH_SIZE = 64
BATCH_INDEX = 0

# prepare fashion_mnist dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data() 
x_train = np.reshape(x_train, [-1, 28, 28, 1])
x_test = np.reshape(x_test, [-1, 28, 28, 1])
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

# show an example
# plt.imshow(x_train[1].reshape((28, 28)))
# plt.title(np.argmax(y_train[1]))
# plt.show()

# build cnn neural network
model = Sequential()
# conv1
model.add(Convolution2D(
    batch_input_shape = (None, 28, 28, 1),
    filters=16,
    kernel_size=3,
    strides=1,
    padding='same',
    data_format='channels_last',
    kernel_regularizer=keras.regularizers.l2(l=0.01)
    ))
model.add(Activation('relu'))
model.add(MaxPooling2D(
    pool_size=2,
    strides=2,
    padding='same',
    data_format='channels_last'
    ))  # output = (28, 28, 32)
# conv2
model.add(Convolution2D(32, 3, strides=1, padding='same', data_format='channels_last'))
model.add(Activation('relu'))
model.add(MaxPooling2D(2, 2, data_format='channels_last')) # output = (14, 14, 64)

# conv3
model.add(Convolution2D(64, 3, strides=1, padding='same', data_format='channels_last'))
model.add(Activation('relu'))
model.add(MaxPooling2D(2, 2, data_format='channels_last')) # output = (7, 7, 128)

# fully connected layers 1 input (7, 7, 128) output (1024)
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))

# fully connected layers 2 input (1024) output (10)
model.add(Dense(10 ,activation='softmax'))
#model.add(Activation('softmax'))

# compile learning process
rmsprop = optimizers.RMSprop()
model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy']
    )


def generator(features, labels, batch_size):

     # Create empty arrays to contain batch of features and labels#
     batch_features = np.zeros((batch_size, 28, 28, 1))
     batch_labels = np.zeros((batch_size,10))

     while True:
       for i in range(batch_size):
         # choose random index in features
         index= np.random.choice(len(features),1)
         batch_features[i] = features[index]
         batch_labels[i] = labels[index]
       yield batch_features, batch_labels

# start training
# 暂时不知道怎么控制验证的间隔
# model.fit(x_train, y_train, batch_size=32, epochs=2, shuffle=True, validation_data=(x_test, y_test))
# verbose Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
model.fit_generator(generator(x_train, y_train, 32), epochs=20, steps_per_epoch=200, validation_data=(x_test, y_test), validation_steps=100)

# 人为控制验证，batch
# for step in range(2001):
#     # data shape = (batch_num, steps, inputs/outputs)
#     X_batch = x_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :, :]
#     Y_batch = y_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :]
#     loss, accuracy = model.train_on_batch(X_batch, Y_batch)
#     BATCH_INDEX += BATCH_SIZE
#     BATCH_INDEX = 0 if BATCH_INDEX >= x_train.shape[0] else BATCH_INDEX
#     print('[{0}/4001]train loss:{1:.4f}, accuracy:{2:.4f}'.format(step, loss, accuracy))
#     if step % 50 == 0:
#         print('----- evaluating -----')
#         loss, accuracy = model.evaluate(x_test, y_test, batch_size=1000)
#         print('valid loss:{0:.4f}, accuracy:{1:.4f}'.format(loss, accuracy))
#         time.sleep(0.1)


# start testing
loss, accuracy = model.evaluate(x_test, y_test)
print('loss:{0:.4f}, accuracy:{1:.4f}'.format(loss, accuracy))

# predict
pred = model.predict(x_test[20:40])
print('real label:{0}'.format(np.argmax(y_test[20:40], axis=1)))
print('  predict :{0}'.format(np.argmax(pred, axis=1)))

keras.backend.clear_session()