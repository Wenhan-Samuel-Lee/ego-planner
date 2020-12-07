import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, Model
from tensorflow.keras.models import load_model
from UAVSequence import UAVSequence, ImageSequence
# from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, concatenate, Flatten, Reshape
# from tensorflow.keras.optimizers import SGD
# from tensorflow.keras.regularizers import l2
import pandas as pd
import numpy as np
import math
from PIL import Image

import rospy
from std_msgs.msg import String
from quadrotor_msgs.msg import PositionCommand
from sensor_msgs.msg import Image as SensorImage
from geometry_msgs.msg import PoseStamped, Point, Vector3
from nav_msgs.msg import Odometry
# https://answers.ros.org/question/326226/importerror-dynamic-module-does-not-define-module-export-function-pyinit__tf2/
from tf.transformations import euler_from_quaternion, quaternion_from_euler
# https://medium.com/@beta_b0t/how-to-setup-ros-with-python-3-44a69ca36674
import message_filters
import cv2
from cv_bridge import CvBridge, CvBridgeError
import csv


# NUM_INPUTS = 13
# NUM_OUTPUTS = 9

# img_shape = (480, 640)

# Load the dataset input and output
# x = np.loadtxt('/home/pranavr/ego-planner/src/neural_net/dataset/0_input.csv', delimiter=',')#, usecols=tuple(range(NUM_INPUTS))) # (num training points, 13)
# y = np.loadtxt('/home/pranavr/ego-planner/src/neural_net/dataset/0_output.csv', delimiter=',', usecols=tuple(range(NUM_OUTPUTS))) # (num training points, 11)
# x = x[:100]
# y=  y[:100]
# # Shuffle x and y, but store the indices so we can still match them with the images
# hash_table = np.random.permutation(len(x))

# x = x[hash_table]
# y = y[hash_table]
# VALIDATION_SPLIT = 0.2
# BATCH_SIZE = 32
# xy = UAVSequence(x, y, BATCH_SIZE, img_shape, hash_table, VALIDATION_SPLIT, False)

# https://jacqueskaiser.com/posts/2020/03/ros-tensorflow
graph = tf.get_default_graph()
session = tf.compat.v1.keras.backend.get_session()

model_complete = tf.keras.models.load_model('/home/pranavr/ego-planner/src/uav_simulator/recorder/models/complete.tf')


# print(xy[1][1][0])
# single_x_test = xy[1][0]
# single_x_test[0] = single_x_test[0][0]
# single_x_test[1] = single_x_test[1][0]
# print(single_x_test[0].shape)
# print(single_x_test[1].shape)
# single_x_test[0] = np.array([single_x_test[0],])
# single_x_test[1] = np.array([single_x_test[1],])
# print(single_x_test[0].shape)
# print(single_x_test[1].shape)

# print(len(single_x_test))
# prediction = model_complete.predict(single_x_test)
# print(prediction)

# exit()

id = 0
current_goal = {}

goal_set = False

# model_complete._make_predict_function()

# session = keras.backend.get_session()
# init = tf.global_variables_initializer()
# session.run(init)

# graph = tf.get_default_graph()

# session = tf.Session()
# tf.keras.backend.set_session(session)
# weights = model_complete.get_weights()
# single_item_model = tf.create_model(batch_size=1)
# single_item_model.set_weights(weights)
# single_item_model.compile()

# model_complete = single_item_model

pos_cmd_publisher = rospy.Publisher('/planning/pos_cmd', PositionCommand, queue_size=10)


def goalCallback(goal):
	global goal_set
	global current_goal
	
	current_goal = goal.pose
	rospy.loginfo("[GOAL CMD: [%f, %f, %f], [%f, %f, %f, %f]\n", 
		goal.pose.position.x,
	    goal.pose.position.y,
	    goal.pose.position.z,

	    goal.pose.orientation.x,
	    goal.pose.orientation.y,
	    goal.pose.orientation.z,
	    goal.pose.orientation.w
		)

	goal_set = True


def callback(img, odom):

	global goal_set
	global current_goal

	if not goal_set:
		rospy.loginfo("[POS CMD: [%f, %f, %f]\n", 
				odom.pose.pose.position.x,
				odom.pose.pose.position.y,
				odom.pose.pose.position.z,
			)
		

	if goal_set:
		print(current_goal)
		orientation = [
			odom.pose.pose.orientation.x,
			odom.pose.pose.orientation.y,
			odom.pose.pose.orientation.z,
			odom.pose.pose.orientation.w
		]
		rospy.loginfo("[POS CMD: [%f, %f, %f], [%f, %f, %f, %f]\n", 
				odom.pose.pose.position.x,
				odom.pose.pose.position.y,
				odom.pose.pose.position.z,
				
				odom.pose.pose.orientation.x,
				odom.pose.pose.orientation.y,
				odom.pose.pose.orientation.z,
				odom.pose.pose.orientation.w
			)
		
		roll, pitch, yaw = euler_from_quaternion(orientation)
		rel_x, rel_y = current_goal.position.x - odom.pose.pose.position.x, \
					current_goal.position.y - odom.pose.pose.position.y
		angle = math.atan2(rel_y, rel_x) - yaw
		distance = math.sqrt(rel_y * rel_y + rel_x * rel_x)
		distance = min(2, distance)

		odom_input = np.asarray([
			distance,
			angle,
			roll,
			pitch,
			yaw,

			odom.twist.twist.linear.x, 
			odom.twist.twist.linear.y,
			odom.twist.twist.linear.z,
			odom.twist.twist.angular.x, 
			odom.twist.twist.angular.y,
			odom.twist.twist.angular.z  
		])


		# might add cvbridge
		bridge = CvBridge()
		cv_image = bridge.imgmsg_to_cv2(img, "32FC1")
		cv_image = cv2.convertScaleAbs(cv_image, alpha=30)
		# print(cv_image.size)
		cv2.imshow("Image window", cv_image)
		cv2.waitKey(3)

		odom_input = np.asarray([
			rel_x,
			rel_y,
			odom.pose.pose.position.z,
			
			odom.pose.pose.orientation.x,
			odom.pose.pose.orientation.y,
			odom.pose.pose.orientation.z,
			odom.pose.pose.orientation.w,

			odom.twist.twist.linear.x, 
			odom.twist.twist.linear.y,
			odom.twist.twist.linear.z,
			odom.twist.twist.angular.x, 
			odom.twist.twist.angular.y,
			odom.twist.twist.angular.z  
		])

		# rospy.
		# https://stackoverflow.com/questions/54652536/keras-tensorflow-backend-error-tensor-input-10-specified-in-either-feed-de
		# with session.as_default():
		# with graph.as_default():
		# odom_input = odom_input.reshape(1, len(odom_input))
		cv_image = np.asarray(cv_image).reshape(480, 640, 1) / 255
		# cv_image = np.expand_dims(cv_image, axis=0)
		print(cv_image.shape)
		print(odom_input.shape)
		print()
		cv_image = np.array([cv_image,])
		odom_input = np.array([odom_input,])
		print(cv_image.shape)
		print(odom_input.shape)

		# model_complete = tf.keras.models.load_model('/home/pranavr/ego-planner/src/uav_simulator/recorder/models/complete.tf')
		# with graph.as_default():
		with session.graph.as_default():
			tf.compat.v1.keras.backend.set_session(session)
			pos_cmd = model_complete.predict([cv_image, odom_input])

			global id
			print(pos_cmd.shape)
			pos_cmd = pos_cmd[0]
			pos_cmd = PositionCommand(None, Point(*pos_cmd[0:3]), Vector3(*pos_cmd[3:6]), Vector3(*pos_cmd[6:9]), pos_cmd[9], pos_cmd[10], [0, 0, 0], [0, 0, 0], id, PositionCommand.TRAJECTORY_STATUS_READY)
			pos_cmd_publisher.publish(pos_cmd)
			rospy.loginfo("[POS CMD: pos_cmd\ {} n".format(pos_cmd))
		id += 1

rospy.init_node('playback_node')

rospy.loginfo("started node\n")

image_sub = message_filters.Subscriber('/pcl_render_node/depth', SensorImage)
# pos_cmd_sub = message_filters.Subscribe('/planning/pos_cmd', PositionCommand)
odom_sub = message_filters.Subscriber('/visual_slam/odom', Odometry)

# rospy.Subscriber("/visual_slam/odom", Odometry, callback, buff_size=10)
rospy.Subscriber("/move_base_simple/goal", PoseStamped, goalCallback, buff_size=10)

# pos_cmd_publisher = rospy.Publisher('/planning/pos_cmd', PositionCommand, queue_size=10)
# pos_cmd_publisher = rospy.Publisher('/planning/pos_cmd')
# pos_cmd_publisher = rospy.Publisher('/planning/pos_cmd')
# pos_cmd_publisher = rospy.Publisher('/planning/pos_cmd')
# pos_cmd_publisher = rospy.Publisher('/planning/pos_cmd')
# pos_cmd_publisher = rospy.Publisher('/planning/pos_cmd')

# ts = message_filters.ApproximateTimeSynchronizer([image_sub, odom_sub], 5, 0.1, allow_headerless=True)
ts = message_filters.ApproximateTimeSynchronizer([image_sub, odom_sub], 100, 1)
ts.registerCallback(callback)

rospy.spin()