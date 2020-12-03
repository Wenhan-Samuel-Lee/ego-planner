import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.models import load_model
# from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, concatenate, Flatten, Reshape
# from tensorflow.keras.optimizers import SGD
# from tensorflow.keras.regularizers import l2
import pandas as pd
import numpy as np
from PIL import Image

import rospy
from std_msgs.msg import String
from quadrotor_msgs.msg import PositionCommand
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import message_filters
import cv2
from cv_bridge import CvBridge, CvBridgeError
import csv

id = 0
current_goal = {}

goal_set = False;

def goalCallback(goal):
	current_goal = goal.pose
	# rospy.loginfo("[GOAL CMD: [%f, %f, %f], [%f, %f, %f, %f]\n", 
	# 	goal.pose.position.x,
	#     goal.pose.position.y,
	#     goal.pose.position.z,

	#     goal.pose.orientation.x,
	#     goal.pose.orientation.y,
	#     goal.pose.orientation.z,
	#     goal.pose.orientation.w
	# 	)

	goal_set = True


def callback(odom):
	rospy.loginfo("[POS CMD: [%f, %f, %f], [%f, %f, %f, %f]\n", 
	    odom.pose.pose.position.x,
	    odom.pose.pose.position.y,
	    odom.pose.pose.position.z,
	    
	    odom.pose.pose.orientation.x,
	    odom.pose.pose.orientation.y,
	    odom.pose.pose.orientation.z,
	    odom.pose.pose.orientation.w
	)

	# might add cvbridge


rospy.init_node('playback_node')

rospy.loginfo("started node\n")

# image_sub = message_filters.Subscribe('/pcl_render_node/depth', Image)
# pos_cmd_sub = message_filters.Subscribe('/planning/pos_cmd', PositionCommand)
# odom_sub = message_filters.Subscribe('/visual_slam/odom', Odometry)

rospy.Subscriber("/visual_slam/odom", Odometry, callback, buff_size=10)
rospy.Subscriber("/move_base_simple/goal", PoseStamped, goalCallback, buff_size=10)

# pos_cmd_publisher = rospy.Publisher('/planning/pos_cmd')

# ts = message_filters.ApproximateTimeSynchronizer([image_sub, odom_sub], 5, 0.1, allow_headerless=True)
# ts = message_filters.ApproximateTimeSynchronizer([odom_sub], 5, 0.1, allow_headerless=True)
# ts.registerCallback(callback)

rospy.spin()