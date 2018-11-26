
# import os ; os.environ['HDF5_DISABLE_VERSION_CHECK']='2'
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Flatten, Dropout
from keras.utils import to_categorical
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn import datasets
# Model
model = Sequential()

model.add(Conv2D(16, filter_shape=(3,3), strides=1, input_shape=(1,8,8)))
model.add(Activation('relu'))
model.add(Dropout(rate = 0.2))
model.add(Conv2D(n_filters=32, filter_shape=(3,3), strides=1, padding='same'))
model.add(Activation('relu'))
model.add(Dropout(rate = 0.2))
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# Data Preprocessing
mnist = fetch_mldata('MNIST original')
X = mnist.data
y = mnist.target

X = (X.astype(np.float32) - 127.5) / 127.5

data = datasets.load_digits()
X = data.data
y = data.target

y = to_categorical(y, num_classes = 10)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train = X_train.reshape((-1,1,8,8))
X_test = X_test.reshape((-1,1,8,8))

model.fit(X_train, y_train, epochs=10, batch_size=32)
# model.fit(x_train, y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=1,
#           validation_data=(x_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])