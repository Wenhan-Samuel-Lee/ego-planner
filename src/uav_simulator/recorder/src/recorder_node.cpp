#include "ros/ros.h"
#include "std_msgs/String.h"
#include "quadrotor_msgs/PositionCommand.h"

#include <image_transport/image_transport.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

/**
 * This tutorial demonstrates simple receipt of messages over the ROS system.
 */
void depthmapCallback(const sensor_msgs::ImageConstPtr& msg)
{
  // ROS_INFO("received callback\n");
  cv_bridge::CvImageConstPtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvShare(msg, "32FC1");
    cv::Mat mono8_img = cv::Mat(cv_ptr->image.size(), CV_8UC1);
    cv::convertScaleAbs(cv_ptr->image, mono8_img, 100, 0.0);
    cv::imshow("view", mono8_img);
    cv::waitKey(20);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("Could not convert from '%s' to '32FC1'.", msg->encoding.c_str());
  }
}

void outputCallback(const quadrotor_msgs::PositionCommand& cmd) {
  ROS_INFO("[POS CMD %d: [%f, %f, %f], [%f, %f, %f], [%f, %f, %f], yaw: %f, %f\n", 
    // cmd.header.stamp,
    cmd.trajectory_id,

    cmd.position.x,
    cmd.position.y,
    cmd.position.z,

    cmd.velocity.x,
    cmd.velocity.y,
    cmd.velocity.z,

    cmd.acceleration.x,
    cmd.acceleration.y,
    cmd.acceleration.z,

    cmd.yaw,
    cmd.yaw_dot
  );
}

void callback(const sensor_msgs::ImageConstPtr& msg, const quadrotor_msgs::PositionCommand& cmd) {

  // Write the contents of these two messages to a CSV file for training
  // Right now, just use every 100 pixels arbitrarily in the image, later on we need to run all of through some other sort of filter
  // Want to use two CSV files, one for input and one for output
  std::ofstream outputFile;
  std::ofstream imageFile;
  
  outputFile.open("output.csv");
  outputFile << itoa(cmd.position.x) + "," + itoa(cmd.position.y) + "," + itoa(cmd.position.z) + "," + 
            itoa(cmd.velocity.x) + "," + itoa(cmd.velocity.y) + "," + itoa(cmd.velocity.z) + "," + 
            itoa(cmd.acceleration.x) + "," + itoa(cmd.acceleration.y) + "," + itoa(cmd.acceleration.z) + "," + 
            itoa(cmd.yaw) + "," + itoa(cmd.yaw_dot) + "," + itoa(cmd.kx[0]) + "," + itoa(cmd.kx[1]) + "," + 
            itoa(cmd.kx[2]) + "," + itoa(cmd.kv[0]) + "," + itoa(cmd.kv[1]) + "," + itoa(cmd.kv[2]) + "\n";
  outputFile.close();

  imageFile.open("image.csv");
  int arraySize = sizeof(msg.data) / sizeof(msg.data[0]);
  for(int i = 0; i < arraySize - 100; i += 100) {
    imageFile << itoa(msg.data[i]) + ",";
  }
  imageFile << itoa(msg.data[arraySize - 1]) + "\n";
  imageFile.close();

  return 0;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "recorder");

  ros::NodeHandle n;

  cv::namedWindow("view");
  // cv::startWindowThread();
  
  image_transport::ImageTransport it(n);
  ROS_INFO("started node\n");
  image_transport::Subscriber sub = it.subscribe("/pcl_render_node/depth", 1000, depthmapCallback);

  ros::Subscriber outputSup = n.subscribe("/planning/pos_cmd", 100, outputCallback);

  message_filters::Subscriber<Image> = image_sub(n, "/pcl_render_node/depth", 1);
  message_filters::Subscriber<PositionCommand> = pos_cmd_sub(n, "/planning/pos_cmd", 1);
  typedef sync_policies::ApproximateTime<Image, PositionCommand> policy;
  Synchronizer<policy> sync(policy(10), image_sub, pos_cmd_sub);
  sync.registerCallback(boost::bind(&callback, _1, _2));

  ros::spin();

  return 0;
}