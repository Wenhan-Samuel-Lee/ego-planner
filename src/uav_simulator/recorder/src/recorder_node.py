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
dataset_folder = "dataset"
input_data = dataset_folder + "/0_input.csv"
output_data = dataset_folder + "/0_output.csv"
output_imgs = dataset_folder + "/img_"
output_ext = ".jpg"
current_goal = {}

save = True;
goal_set = False;

def goalCallback(goal):
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

def callback(msg, cmd, odom):
	rospy.loginfo("[POS CMD: [%f, %f, %f], [%f, %f, %f, %f]\n", 
	    odom.pose.pose.position.x,
	    odom.pose.pose.position.y,
	    odom.pose.pose.position.z,
	    
	    odom.pose.pose.orientation.x,
	    odom.pose.pose.orientation.y,
	    odom.pose.pose.orientation.z,
	    odom.pose.pose.orientation.w
	    )

	if save and goal_set and odom != None:
		with open(output_data, mode='w') as output_file:
			output_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
			output_writer.writerow([
				str(cmd.position.x), str(cmd.position.y), str(cmd.position.z), 
				str(cmd.velocity.x), str(cmd.velocity.y), str(cmd.velocity.z), 
				str(cmd.acceleration.x), str(cmd.acceleration.y), str(cmd.acceleration.z), 
				str(cmd.yaw), str(cmd.yaw_dot)
				])
		with open(input_data, mode='w') as input_file:
			input_writer = csv.writer(input_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
			input_writer.writerow([
				str(current_goal.position.x - odom.pose.pose.position.x), 
				str(current_goal.position.y - odom.pose.pose.position.y), 
				str(odom.pose.pose.position.z), 
				str(odom.pose.pose.orientation.x), 
				str(odom.pose.pose.orientation.y), 
				str(odom.pose.pose.orientation.z), 
				str(odom.pose.pose.orientation.w), 
				str(odom.twist.twist.linear.x), 
				str(odom.twist.twist.linear.y), 
				str(odom.twist.twist.linear.z), 
				str(odom.twist.twist.angular.x), 
				str(odom.twist.twist.angular.y), 
				str(odom.twist.twist.angular.z), 
				])

	# might add cvbridge


rospy.init_node('recorder')

rospy.loginfo("started node\n")

image_sub = message_filters.Subscribe('/pcl_render_node/depth', Image)
pos_cmd_sub = message_filters.Subscribe('/planning/pos_cmd', PositionCommand)
odom_sub = message_filters.Subscribe('/visual_slam/odom', Odometry)

rospy.Subscriber("/move_base_simple/goal", PoseStamped, goalCallback)

ts = message_filters.ApproximateTimeSynchronizer([image_sub, pos_cmd_sub, odom_sub], 10, 0.1, allow_headerless=True)
ts.registerCallback(callback)

rospy.spin()