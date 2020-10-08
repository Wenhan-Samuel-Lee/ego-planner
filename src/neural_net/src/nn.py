import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

model = Sequential()
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=10))

train_X = [[1] * 100] * 100
test_X = [[1] * 100] * 100

train_Y = [[1] * 10] * 100
test_Y = [[1] * 10] * 100

# print(test_Y)

opt = SGD(lr = .1, momentum = 0.9)

model.compile(loss='mean_squared_error',
    optimizer=opt,
    metrics=['mean_squared_error'])

model.fit(train_X, train_Y, epochs=100, batch_size=32)

loss_and_metrics = model.evaluate(test_X, test_Y, batch_size=32)