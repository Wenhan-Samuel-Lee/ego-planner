#include "ros/ros.h"
#include "std_msgs/String.h"
#include "quadrotor_msgs/PositionCommand.h"
#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>

int id = 0;
std::string dataset_folder = "dataset";
std::string input_data = dataset_folder + "/0_input.csv";
std::string output_data = dataset_folder + "/0_output.csv";
std::string output_imgs = dataset_folder + "/img_";
std::string output_ext = ".jpg";

geometry_msgs::Pose current_goal{};

std::ofstream inputFile;
std::ofstream outputFile;

bool save = true;
bool goal_set = false;

/**
 * This tutorial demonstrates simple receipt of messages over the ROS system.
 */
// void depthmapCallback(const sensor_msgs::ImageConstPtr& msg)
// {
//   ROS_INFO("received callback\n");
//   cv_bridge::CvImageConstPtr cv_ptr;
//   try
//   {
//     cv_ptr = cv_bridge::toCvShare(msg, "32FC1");
//     cv::Mat mono8_img = cv::Mat(cv_ptr->image.size(), CV_8UC1);
//     cv::convertScaleAbs(cv_ptr->image, mono8_img, 100, 0.0);
//     cv::imshow("view", mono8_img);

//     // ROS_INFO("writing image\n");
//     // cv:imwrite("id" + std::to_string(id++) + ".jpg", mono8_img);

//     cv::waitKey(20);
//   }
//   catch (cv_bridge::Exception& e)
//   {
//     ROS_ERROR("Could not convert from '%s' to '32FC1'.", msg->encoding.c_str());
//   }
// }

// void outputCallback(const quadrotor_msgs::PositionCommand& cmd) {
//   ROS_INFO("[POS CMD %d: [%f, %f, %f], [%f, %f, %f], [%f, %f, %f], yaw: %f, %f\n", 
//     // cmd.header.stamp,
//     cmd.trajectory_id,

//     cmd.position.x,
//     cmd.position.y,
//     cmd.position.z,

//     cmd.velocity.x,
//     cmd.velocity.y,
//     cmd.velocity.z,

//     cmd.acceleration.x,
//     cmd.acceleration.y,
//     cmd.acceleration.z,

//     cmd.yaw,
//     cmd.yaw_dot
//   );
// }

void goalCallback(const geometry_msgs::PoseStamped& goal) {
  current_goal = goal.pose;
  ROS_INFO("[GOAL CMD: [%f, %f, %f], [%f, %f, %f, %f]\n", 
    // cmd.header.stamp,
  
    goal.pose.position.x,
    goal.pose.position.y,
    goal.pose.position.z,

    goal.pose.orientation.x,
    goal.pose.orientation.y,
    goal.pose.orientation.z,
    goal.pose.orientation.w
  );

  goal_set = true;
}

void callback(const sensor_msgs::ImageConstPtr& msg, const quadrotor_msgs::PositionCommandConstPtr& cmd, const nav_msgs::Odometry::ConstPtr& odom) {
    // ROS_INFO("callback called\n");
  // Write the contents of these two messages to a CSV file for training
  // Right now, just use every 100 pixels arbitrarily in the image, later on we need to run all of through some other sort of filter
  // Want to use two CSV files, one for input and one for output
  // std::ofstream imageFile;
  
  ROS_INFO("[POS CMD: [%f, %f, %f], [%f, %f, %f, %f]\n", 
    odom->pose.pose.position.x,
    odom->pose.pose.position.y,
    odom->pose.pose.position.z,
    
    odom->pose.pose.orientation.x,
    odom->pose.pose.orientation.y,
    odom->pose.pose.orientation.z,
    odom->pose.pose.orientation.w);

  
  if (save && goal_set && odom != nullptr) {
    outputFile << 
            std::to_string(cmd->position.x) + "," + std::to_string(cmd->position.y) + "," + std::to_string(cmd->position.z) + "," + 
            std::to_string(cmd->velocity.x) + "," + std::to_string(cmd->velocity.y) + "," + std::to_string(cmd->velocity.z) + "," + 
            std::to_string(cmd->acceleration.x) + "," + std::to_string(cmd->acceleration.y) + "," + std::to_string(cmd->acceleration.z) + "," + 
            std::to_string(cmd->yaw) + "," + std::to_string(cmd->yaw_dot) + "\n";
  
    inputFile << 
            std::to_string(current_goal.position.x) + "," + 
            std::to_string(current_goal.position.y) + "," +
            std::to_string(odom->pose.pose.position.x) + "," + 
            std::to_string(odom->pose.pose.position.y) + "," +
            std::to_string(odom->pose.pose.position.z) + "," +
            std::to_string(odom->pose.pose.orientation.x) + "," + 
            std::to_string(odom->pose.pose.orientation.y) + "," +
            std::to_string(odom->pose.pose.orientation.z) + "," +
            std::to_string(odom->pose.pose.orientation.w) + "," +
            std::to_string(odom->twist.twist.linear.x) + "," + 
            std::to_string(odom->twist.twist.linear.y) + "," +
            std::to_string(odom->twist.twist.linear.z) + "," +
            std::to_string(odom->twist.twist.angular.x) + "," + 
            std::to_string(odom->twist.twist.angular.y) + "," +
            std::to_string(odom->twist.twist.angular.z) + "\n";
  }

  // ROS_INFO("[POS CMD %d: [%f, %f, %f], [%f, %f, %f], [%f, %f, %f], yaw: %f, %f\n", 
  //   // cmd.header.stamp,
  //   cmd->trajectory_id,

  //   cmd->position.x,
  //   cmd->position.y,
  //   cmd->position.z,

  //   cmd->velocity.x,
  //   cmd->velocity.y,
  //   cmd->velocity.z,

  //   cmd->acceleration.x,
  //   cmd->acceleration.y,
  //   cmd->acceleration.z,

  //   cmd->yaw,
  //   cmd->yaw_dot
  // );

  cv_bridge::CvImageConstPtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvShare(msg, "32FC1");
    cv::Mat mono8_img = cv::Mat(cv_ptr->image.size(), CV_8UC1);
    cv::convertScaleAbs(cv_ptr->image, mono8_img, 30, 0.0);
    // cv::imshow("view", mono8_img);
    // ROS_INFO("writing image\n");

    if (save && goal_set && odom != nullptr)
      cv:imwrite(output_imgs + std::to_string(id++) + output_ext, mono8_img);
    cv::waitKey(20);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("Could not convert from '%s' to '32FC1'.", msg->encoding.c_str());
  }
}

int main(int argc, char **argv)
{
  using namespace message_filters;
  ros::init(argc, argv, "recorder");

  ros::NodeHandle n;

  if (save) {
    inputFile.open(input_data, std::ios::app);
    outputFile.open(output_data, std::ios::app);
  }
  // cv::namedWindow("view");
  // cv::startWindowThread();
  
  // image_transport::ImageTransport it(n);
  ROS_INFO("started node\n");
  // image_transport::Subscriber sub = it.subscribe("/pcl_render_node/depth", 1000, depthmapCallback);

  // ros::Subscriber outputSup = n.subscribe("/planning/pos_cmd", 100, outputCallback);


  message_filters::Subscriber<sensor_msgs::Image> image_sub(n, "/pcl_render_node/depth", 10);
  message_filters::Subscriber<quadrotor_msgs::PositionCommand> pos_cmd_sub(n, "/planning/pos_cmd", 10);
  message_filters::Subscriber<nav_msgs::Odometry> odom_sub(n, "/visual_slam/odom", 10);

  ros::Subscriber goal_cmd_sub = n.subscribe("/move_base_simple/goal", 10, goalCallback);

  
  typedef sync_policies::ApproximateTime<sensor_msgs::Image, quadrotor_msgs::PositionCommand,nav_msgs::Odometry> Policy;
  message_filters::Synchronizer<Policy> sync(Policy(100), image_sub, pos_cmd_sub, odom_sub);

  // message_filters::TimeSynchronizer<sensor_msgs::Image, quadrotor_msgs::PositionCommand> sync(image_sub, pos_cmd_sub, 100);
  sync.registerCallback(boost::bind(&callback, _1, _2, _3));

  ros::spin();

  outputFile.close();

  return 0;
}