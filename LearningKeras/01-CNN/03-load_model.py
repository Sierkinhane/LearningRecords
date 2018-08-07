import keras
from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Convolution2D, Activation, Flatten
from keras.datasets import fashion_mnist
import keras.optimizers as optimizers
import numpy as np 

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data() 
x_train = np.reshape(x_train, [-1, 28, 28, 1])
x_test = np.reshape(x_test, [-1, 28, 28, 1])
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

model = Sequential()
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
    ))  # output = (28, 28, 16)
# conv2
model.add(Convolution2D(32, 3, strides=1, padding='same', data_format='channels_last'))
model.add(Activation('relu'))
model.add(MaxPooling2D(2, 2, data_format='channels_last')) # output = (28, 28, 32)

# conv3
model.add(Convolution2D(64, 3, strides=1, padding='same', data_format='channels_last'))
model.add(Activation('relu'))
model.add(MaxPooling2D(2, 2, data_format='channels_last')) # output = (7, 7, 64)

# fully connected layers 1 input (7, 7, 64) output (1024)
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))

# fully connected layers 2 input (1024) output (10)
model.add(Dense(10))
model.add(Activation('softmax'))

'''
reload weights and continue training
'''
model.load_weights('./fashion_mnist_model/fashion_model.h5')

rmsprop = optimizers.adam(lr=0.0001, decay=0.0005)
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

model.fit_generator(generator(x_train, y_train, 32), epochs=20, steps_per_epoch=200, validation_data=(x_test, y_test), validation_steps=100)


pred = model.predict(x_test[20:40])
print('real label:{0}'.format(np.argmax(y_test[20:40], axis=1)))
print('  predict :{0}'.format(np.argmax(pred, axis=1)))

# weights = model.get_weights()
# print(weights)
# yaml = model.to_yaml()
# print(yaml)

keras.backend.clear_session()