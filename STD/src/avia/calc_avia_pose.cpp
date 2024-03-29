#include <nav_msgs/Odometry.h>
#include <pcl/common/io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud2.h>
#include <sstream>
#include <std_msgs/Header.h>
#include <stdio.h>
#include <tf/tf.h>
#include <tf_conversions/tf_eigen.h>

int main(int argc, char **argv) {
  ros::init(argc, argv, "reshape_data");
  ros::NodeHandle nh;
  std::string bag_path = "";
  std::string pose_path = "";
  double icp_threshold = 0.5;
  nh.param<std::string>("bag_path", bag_path, "");
  nh.param<std::string>("pose_path", pose_path, "");
  std::ofstream pose_file(pose_path);

  getchar();
  // evaluate time
  int msg_number = 0;
  std::fstream file_;
  file_.open(bag_path, std::ios::in);
  if (!file_) {
    std::cout << "File " << bag_path << " does not exit" << std::endl;
  }
  ROS_INFO("Start to load the rosbag %s", bag_path.c_str());
  rosbag::Bag bag;
  try {
    bag.open(bag_path, rosbag::bagmode::Read);
  } catch (rosbag::BagException e) {
    ROS_ERROR_STREAM("LOADING BAG FAILED: " << e.what());
  }
  std::vector<std::string> types;
  types.push_back(std::string("sensor_msgs/PointCloud2"));
  types.push_back(std::string("nav_msgs/Odometry"));
  rosbag::View view(bag, rosbag::TypeQuery(types));
  int count = 0;
  pcl::PointCloud<pcl::PointXYZI> cloud;
  nav_msgs::Odometry odom;
  Eigen::Vector3d translation;
  Eigen::Matrix3d rot;
  Eigen::Quaterniond q;
  bool cloud_ready = false;
  bool pose_ready = false;
  long laser_time = 0;
  long odom_time = 0;
  pcl::PointCloud<pcl::PointXYZI>::Ptr current_cloud(
      new pcl::PointCloud<pcl::PointXYZI>);
  BOOST_FOREACH (rosbag::MessageInstance const m, view) {
    std::time_t startTime = clock();
    sensor_msgs::PointCloud2::ConstPtr cloud_ptr =
        m.instantiate<sensor_msgs::PointCloud2>();
    if (cloud_ptr != NULL) {
      laser_time = cloud_ptr->header.stamp.toNSec();
      cloud_ready = true;
    }
    nav_msgs::Odometry::ConstPtr odom_ptr = m.instantiate<nav_msgs::Odometry>();
    if (odom_ptr != NULL) {
      // std::cout << "load pose" << std::endl;
      odom_time = odom_ptr->header.stamp.toNSec();
      pose_ready = true;
      translation << odom_ptr->pose.pose.position.x,
          odom_ptr->pose.pose.position.y, odom_ptr->pose.pose.position.z;
      q.x() = odom_ptr->pose.pose.orientation.x;
      q.y() = odom_ptr->pose.pose.orientation.y;
      q.z() = odom_ptr->pose.pose.orientation.z;
      q.w() = odom_ptr->pose.pose.orientation.w;
      // std::cout << "time_inc:" << odom_time - laser_time << std::endl;
    }
    if (cloud_ready && pose_ready) {
      pose_file << laser_time << "," << translation[0] << "," << translation[1]
                << "," << translation[2] << "," << q.w() << "," << q.x() << ","
                << q.y() << "," << q.z() << std::endl;
      cloud_ready = false;
      pose_ready = false;
      if (msg_number % 100 == 0) {
        std::cout << "msg number:" << msg_number << std::endl;
      }
      msg_number++;
    }
  }
  return 0;
}