import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import rosbag
import pandas as pd

bag = rosbag.Bag('./') #bag file
topic = #topic
column_names = 
df = pd.DataFrame(columns=column_names)
for topic, msg, t in bag.read_messages(topics=topic):
	x = msg.x
	y = msg.y
	df = df.append(
	)

df.to_csv('out.csv')

# model = Sequential()
# model.add(Dense(units=100, activation='relu'))
# model.add(Dense(units=10))

# train_X = [[1] * 100] * 100
# test_X = [[1] * 100] * 100

# train_Y = [[1] * 10] * 100
# test_Y = [[1] * 10] * 100

# # print(test_Y)

# opt = SGD(lr = .1, momentum = 0.9)

# model.compile(loss='mean_squared_error',
#     optimizer=opt,
#     metrics=['mean_squared_error'])

# model.fit(train_X, train_Y, epochs=100, batch_size=32)

# loss_and_metrics = model.evaluate(test_X, test_Y, batch_size=32)