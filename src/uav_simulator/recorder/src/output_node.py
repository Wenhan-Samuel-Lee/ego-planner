import rospy
from quadrotor_msgs.msg import PositionCommand

def output():
	pub = rospy.Publisher('/planning/pos_cmd', PositionCommand, queue_size=10)
	rospy.init_node('output')
	rate = rospy.Rate(10)
	while not rospy.is_shutdown():
		pub.publish(???)
		rate.sleep()

if __name__ == '__main__':
    try:
        output()
    except rospy.ROSInterruptException:
        pass