#include "ros/ros.h"
#include "std_msgs/String.h"

#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

/**
 * This tutorial demonstrates simple receipt of messages over the ROS system.
 */
void depthmapCallback(const sensor_msgs::ImageConstPtr& msg)
{
  ROS_INFO("received callback\n");
  try
  {
    cv::imshow("view", cv_bridge::toCvShare(msg, "bgr8")->image);
    cv::waitKey(0);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
  }
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "recorder");

  ros::NodeHandle n;

  cv::namedWindow("view");
  cv::startWindowThread();
  
  image_transport::ImageTransport it(n);
  ROS_INFO("started node\n");
  image_transport::Subscriber sub = it.subscribe("/pcl_render_node/depth", 1000, depthmapCallback);

  ros::spin();

  return 0;
}