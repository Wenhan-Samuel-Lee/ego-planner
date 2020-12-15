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
import transforms3d
import os

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


os.system("rosnode kill traj_server")
# def model_loss(target, predicted):
#     print(target.shape)
#     print(predicted.shape)
#     position_loss = tf.keras.losses.MSE(target[:,:3], predicted[:,:3])
#     other_loss = tf.keras.losses.MSE(target[:,3:], predicted[:,3:])
    
#     total_loss = 10 * position_loss + other_loss
#     return total_loss

def pos_metric(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference[:3], axis=-1)

def vel_metric(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference[3:6], axis=-1)

def acc_metric(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference[6:9], axis=-1)

def yaw_metric(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference[9:], axis=-1)
	
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


model_complete = tf.keras.models.load_model('/home/pranavr/ego-planner/src/uav_simulator/recorder/models/complete_empty_30_min_no_image_0_050.tf',
											custom_objects={"pos_metric": pos_metric, "vel_metric": vel_metric, "acc_metric": acc_metric, "yaw_metric": yaw_metric})#, custom_objects={"model_loss": model_loss})

# model_complete = tf.keras.models.load_model('/home/pranavr/ego-planner/src/uav_simulator/recorder/models/complete_empty_30_min_transformed_0_034.tf',
# 											custom_objects={"pos_metric": pos_metric, "vel_metric": vel_metric, "acc_metric": acc_metric, "yaw_metric": yaw_metric})#, custom_objects={"model_loss": model_loss})


# input_mn = [ 0.00071267,  0.00121005, -0.01057223, -0.03100285, -0.04670384, -0.00124892,
#   0.00200639,  0.00305923, -0.01772598]
# input_std = [0.05952703, 0.1066482,  0.47686612, 1.10708861, 1.24799304, 0.12653448,
#  0.37974445, 0.37144707, 0.63460594]
# output_mn = [ 0.00038266, -0.00076545,  0.00295317, -0.03242625, -0.04784536, -0.00096541,
#   0.01030061,  0.00485227, -0.00516944, -0.00201966, -0.02447285]
# output_std = [0.04649806, 0.03959377, 0.00973197, 1.11215682, 1.25204454, 0.12756201,
#  0.62285087, 0.80529237, 0.20998748, 0.06998235, 0.92470723]

input_mn = [ 0.0035498,  -0.02925752, -0.02631224, -0.02401414, -0.00523067, -0.00217564,
 -0.0010349,   0.00572376, -0.03676773]
input_std = [0.16396395, 0.18420277, 0.93733231, 1.20875131, 1.25781257, 0.24468761,
 0.86837934, 0.89024787, 1.27144478]

output_mn = [ 2.65993210e-03,  1.52750180e-03,  1.39032816e-02, -2.26173063e-02,
 -5.06012831e-03,  2.72522870e-05, -1.00951970e-02, -4.60337859e-03,
 -1.37203478e-02, -3.80694860e-03, -5.90423143e-02]
output_std = [0.08561747, 0.09262722, 0.02173819, 1.2201406,  1.27785797, 0.24312548,
 1.43014532, 1.55238503, 0.55457504, 0.14860682, 1.90875224]

# dataset = "/home/pranavr/ego-planner/src/uav_simulator/recorder/dataset/"



def clamp_yaw(diff):
    if diff > math.pi:
        diff -= 2 * math.pi
    elif diff < -1 * math.pi:
        diff += 2 * math.pi
    
    return diff

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

	# if not goal_set:
	# 	rospy.loginfo("[POS CMD: [%f, %f, %f]\n", 
	# 			odom.pose.pose.position.x,
	# 			odom.pose.pose.position.y,
	# 			odom.pose.pose.position.z,
	# 		)
		

	if goal_set:
		# print(current_goal)
		orientation = [
			odom.pose.pose.orientation.x,
			odom.pose.pose.orientation.y,
			odom.pose.pose.orientation.z,
			odom.pose.pose.orientation.w
		]
		# rospy.loginfo("[POS CMD: [%f, %f, %f], [%f, %f, %f, %f]\n", 
		# 		odom.pose.pose.position.x,
		# 		odom.pose.pose.position.y,
		# 		odom.pose.pose.position.z,
				
		# 		odom.pose.pose.orientation.x,
		# 		odom.pose.pose.orientation.y,
		# 		odom.pose.pose.orientation.z,
		# 		odom.pose.pose.orientation.w
		# 	)
		
		roll, pitch, yaw = euler_from_quaternion(orientation)
		rel_x, rel_y = current_goal.position.x - odom.pose.pose.position.x, \
						current_goal.position.y - odom.pose.pose.position.y
		angle = clamp_yaw(math.atan2(rel_y, rel_x) - yaw)
		distance = math.sqrt(rel_y * rel_y + rel_x * rel_x)
		# distance = min(2, distance)
		
		# print(roll, pitch, yaw)
		rot_matrix = np.asarray(transforms3d.euler.euler2mat(roll, pitch, yaw))
		

		# INPUTS
		# linear velocities
		rel_lin_vel = np.matmul(rot_matrix, [odom.twist.twist.linear.x, 
											odom.twist.twist.linear.y,
											odom.twist.twist.linear.z])
		# angular velocities
		rel_ang_vel = np.matmul(rot_matrix, [odom.twist.twist.angular.x, 
											odom.twist.twist.angular.y,
											odom.twist.twist.angular.z])


		odom_input = np.asarray([
			roll,
			pitch,
			angle,
			# distance,
			# yaw,

			*rel_lin_vel,
			*rel_ang_vel 
		])


		# might add cvbridge
		bridge = CvBridge()
		cv_image = bridge.imgmsg_to_cv2(img, "32FC1")
		cv_image = cv2.convertScaleAbs(cv_image, alpha=30)
		# print(cv_image.size)
		# cv2.imshow("Image window", cv_image)
		# cv2.waitKey(3)

		# odom_input = np.asarray([
		# 	-rel_x,
		# 	-rel_y,
		# 	odom.pose.pose.position.z,
			
		# 	odom.pose.pose.orientation.x,
		# 	odom.pose.pose.orientation.y,
		# 	odom.pose.pose.orientation.z,
		# 	odom.pose.pose.orientation.w,

		# 	odom.twist.twist.linear.x, 
		# 	odom.twist.twist.linear.y,
		# 	odom.twist.twist.linear.z,
		# 	odom.twist.twist.angular.x, 
		# 	odom.twist.twist.angular.y,
		# 	odom.twist.twist.angular.z  
		# ])

		# rospy.
		# https://stackoverflow.com/questions/54652536/keras-tensorflow-backend-error-tensor-input-10-specified-in-either-feed-de
		# with session.as_default():
		# with graph.as_default():
		# odom_input = odom_input.reshape(1, len(odom_input))
		cv_image = np.asarray(cv_image).reshape(480, 640, 1) / 255
		# cv_image = np.expand_dims(cv_image, axis=0)
		# print(cv_image.shape)
		# print(odom_input.shape)
		# print()
		cv_image = np.array([cv_image,])
		odom_input = np.array([odom_input,])
		# print(cv_image.shape)
		# print(odom_input.shape)

		# model_complete = tf.keras.models.load_model('/home/pranavr/ego-planner/src/uav_simulator/recorder/models/complete.tf')
		# with graph.as_default():
	
		if distance > 0.5:
			with session.graph.as_default():
				tf.compat.v1.keras.backend.set_session(session)

				# convert = (u - min) / (max - min)
				# for i in range(len(odom_input)):
				# 	odom_input[i] = (odom_input[i] - input_min[i]) / (input_max[i] - input_min[i])
				for i in range(len(odom_input)):
					odom_input[i] = (odom_input[i] - input_mn[i]) / input_std[i]
				# pos_cmd = model_complete.predict([cv_image, odom_input])
				pos_cmd = model_complete.predict(odom_input)

				global id
				# print(pos_cmd.shape)
				rospy.loginfo("[POS CMD: pos_cmd\ {} n".format(pos_cmd))

				# revert = u * (max - min) + min
				pos_cmd = pos_cmd[0]
				for i in range(len(pos_cmd)):
					pos_cmd[i] = (pos_cmd[i] * output_std[i]) + output_mn[i]
				
				rev_rot_matrix = np.linalg.inv(rot_matrix)
				rev_rot_matrix = transforms3d.euler.euler2mat(-roll, -pitch, -yaw)

				print(transforms3d.euler.mat2euler(rot_matrix))
				print(transforms3d.euler.mat2euler(rev_rot_matrix))
				# print(np.around(rev_rot_matrix, 3))
				# rev_rot_matrix = np.transpose(rot_matrix)
				# print(np.around(rev_rot_matrix, 3))
				# rev_rot_matrix = transforms3d.euler.euler2mat(-roll, -pitch, -yaw)
				# print(np.around(rev_rot_matrix, 3))
				# print(np.around(rot_matrix * rev_rot_matrix, 3))
				# pos_cmd[0:3] = [0.1, 0, 0]
				# pos = [0.1, 0, 0]
				pos = np.dot(rev_rot_matrix, pos_cmd[0:3]) * 20
				# pos *= -1
				vel = np.dot(rev_rot_matrix, pos_cmd[3:6])
				# vel *= -1
				acc = np.dot(rev_rot_matrix, pos_cmd[6:9])
				# print(pos.shape, vel.shape, acc.shape)
				# acc[1] *= -1
				# print(np.linalg.matrix_rank(rot_matrix))
				# print(np.linalg.det(rot_matrix))
				# print(transforms3d.euler.mat2euler(rev_rot_matrix))
				# print(np.around(rot_matrix, 3))
				# print(np.around(np.linalg.inv(rev_rot_matrix), 2))
				# print(np.dot(rot_matrix, rev_rot_matrix))

				con = math.pi / 10
				yaw = clamp_yaw(yaw + min(max(-con, pos_cmd[9]), con))
				# yaw = clamp_yaw(math.atan2(rel_y, rel_x))
				pos_cmd = PositionCommand(
					None, 
					Point((pos[0] + odom.pose.pose.position.x),
						  (pos[1] + odom.pose.pose.position.y),
						  (pos[2] + odom.pose.pose.position.z)), 
					# Point(0, 0, 0),
					# Point(odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z), 
					# Vector3(*vel / 5), 
					Vector3(0, 0, 0),
					# Vector3(*acc / 5),
					Vector3(0, 0, 0),
					yaw, 
					# 0,
					# pos_cmd[10] / 5,
					0,
					[0, 0, 0], [0, 0, 0], id, PositionCommand.TRAJECTORY_STATUS_READY)
				
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

# ts = message_filters.ApproximateTimeSynchronizer([image_sub, odom_sub], 5, 0.1, allow_headerless=True)
ts = message_filters.ApproximateTimeSynchronizer([image_sub, odom_sub], 100, 1)
ts.registerCallback(callback)

rospy.spin()