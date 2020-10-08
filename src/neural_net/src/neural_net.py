import numpy as np
np.random.seed(123)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils

model = Sequential()
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1, )))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=32, verbose=1)

score = model.evaluate(X_test, y_test, verbose=0)