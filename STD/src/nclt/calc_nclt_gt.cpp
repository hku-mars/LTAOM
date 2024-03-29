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
  rosbag::View view(bag, rosbag::TypeQuery(types));
  bool is_skip_frame = false;
  int count = 0;
  BOOST_FOREACH (rosbag::MessageInstance const m, view) {
    sensor_msgs::PointCloud2::ConstPtr cloud_ptr =
        m.instantiate<sensor_msgs::PointCloud2>();
    if (cloud_ptr != NULL) {
      if (count == 0) {
        if (!is_skip_frame) {
          pcl::PointCloud<pcl::PointXYZI>::Ptr temp_cloud(
              new pcl::PointCloud<pcl::PointXYZI>);
          key_cloud_list.push_back(temp_cloud);
          current_key_cloud->clear();
        }
      }
      long laser_time = cloud_ptr->header.stamp.toNSec();
      pcl::PCLPointCloud2 pcl_pc;
      pcl::PointCloud<pcl::PointXYZI> pcl_cloud;
      pcl_conversions::toPCL(*cloud_ptr, pcl_pc);
      pcl::fromPCLPointCloud2(pcl_pc, pcl_cloud);
      int pose_index = findPoseIndexUsingTime(time_list, laser_time);
      if (pose_index < 0) {
        is_skip_frame = true;
        continue;
      }
      is_skip_frame = false;
    }
  }

  return 0;
}