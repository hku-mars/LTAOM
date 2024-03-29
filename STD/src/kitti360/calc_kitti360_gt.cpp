#include "include/std.h"
#include <nav_msgs/Odometry.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud2.h>

std::vector<float> read_lidar_data(const std::string lidar_data_path) {
  std::ifstream lidar_data_file(lidar_data_path,
                                std::ifstream::in | std::ifstream::binary);
  lidar_data_file.seekg(0, std::ios::end);
  const size_t num_elements = lidar_data_file.tellg() / sizeof(float);
  lidar_data_file.seekg(0, std::ios::beg);

  std::vector<float> lidar_data_buffer(num_elements);
  lidar_data_file.read(reinterpret_cast<char *>(&lidar_data_buffer[0]),
                       num_elements * sizeof(float));
  return lidar_data_buffer;
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "std_loop");
  ros::NodeHandle nh;
  std::string data_name = "";
  std::string setting_path = "";
  std::string lidar_data_path = "";
  std::string bag_file = "";
  std::string pose_file = "";
  double icp_threshold = 0.5;
  nh.param<std::string>("data_name", data_name, "");
  nh.param<std::string>("setting_path", setting_path, "");
  nh.param<std::string>("lidar_data_path", lidar_data_path, "");
  nh.param<std::string>("pose_file", pose_file, "");
  nh.param<double>("icp_threshold", icp_threshold, 0.5);
  std::string icp_string = std::to_string(icp_threshold);
  // std::string result_path =
  //     "/home/ycj/matlab_code/loop_detection/result/" + data_name + "/" +
  //     data_name + "_" + icp_string.substr(0, icp_string.find(".") + 3) +
  //     ".txt";
  std::ofstream result_file("/home/ycj/matlab_code/loop_detection/gt/"
                            "kitti360_gt/kitti360-04_gt.txt");
  std::ofstream debug_file("/home/ycj/catkin_ws/src/STD/Log/log.txt");
  std::ofstream debug_augment("/home/ycj/catkin_ws/src/STD/Log/augument.txt");

  ros::Publisher pubOdomAftMapped =
      nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 10);
  ros::Publisher pubRegisterCloud =
      nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100);
  ros::Publisher pubCureentCloud =
      nh.advertise<sensor_msgs::PointCloud2>("/cloud_current", 100);
  ros::Publisher pubCurrentBinary =
      nh.advertise<sensor_msgs::PointCloud2>("/cloud_key_points", 100);
  ros::Publisher pubMatchedCloud =
      nh.advertise<sensor_msgs::PointCloud2>("/cloud_matched", 100);
  ros::Publisher pubMatchedBinary =
      nh.advertise<sensor_msgs::PointCloud2>("/cloud_matched_key_points", 100);
  ros::Publisher pubSTD =
      nh.advertise<visualization_msgs::MarkerArray>("descriptor_line", 10);

  ros::Rate loop(50000);

  ConfigSetting config_setting;
  load_config_setting(setting_path, config_setting);

  std::vector<std::pair<Eigen::Vector3d, Eigen::Matrix3d>> pose_list;
  std::vector<double> time_list;
  std::vector<int> frame_number_list;
  load_pose_with_frame(pose_file, pose_list, frame_number_list);
  std::string print_msg = "Sucessfully load pose file:" + pose_file +
                          ". pose size:" + std::to_string(pose_list.size());
  ROS_INFO_STREAM(print_msg.c_str());
  Eigen::Vector3d init_translation = pose_list[0].first;
  for (size_t i = 0; i < pose_list.size(); i++) {
    pose_list[i].first = pose_list[i].first - init_translation;
  }

  // save all point clouds of key frame
  std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> key_cloud_list;

  // save all planes of key frame
  std::vector<pcl::PointCloud<pcl::PointXYZINormal>::Ptr> history_plane_list;

  // save all binary descriptors of key frame
  std::vector<std::vector<BinaryDescriptor>> history_binary_list;

  // save all STD descriptors of key frame
  std::vector<std::vector<STD>> history_STD_list;

  // save all poses(translation only) of key frame
  std::vector<Eigen::Vector3d> key_pose_list;

  // hash table, save all descriptor
  std::unordered_map<STD_LOC, std::vector<STD>> STD_map;

  // calc mean time
  double mean_time = 0;

  // record mean position
  Eigen::Vector3d current_mean_position(0, 0, 0);

  // extrinsic for kitti360
  Eigen::Matrix4d T_cam_to_pose;
  T_cam_to_pose << 0.0371783278, -0.0986182135, 0.9944306009, 1.5752681039,
      0.9992675562, -0.0053553387, -0.0378902567, 0.0043914093, 0.0090621821,
      0.9951109327, 0.0983468786, -0.6500000000, 0, 0, 0, 1;
  Eigen::Matrix4d T_cam_to_lidar;
  T_cam_to_lidar << 0.04307104361, -0.08829286498, 0.995162929, 0.8043914418,
      -0.999004371, 0.007784614041, 0.04392796942, 0.2993489574, -0.01162548558,
      -0.9960641394, -0.08786966659, -0.1770225824, 0, 0, 0, 1;
  Eigen::Matrix4d T_lidar_to_pose;
  T_lidar_to_pose = T_cam_to_pose * T_cam_to_lidar.inverse();
  Eigen::Matrix3d lidar_to_pose_rot = T_lidar_to_pose.block<3, 3>(0, 0);
  Eigen::Vector3d lidar_to_pose_t(T_lidar_to_pose(0, 3), T_lidar_to_pose(1, 3),
                                  T_lidar_to_pose(2, 3));

  // load lidar point cloud and start loop
  int pose_cnt = 0;
  pcl::PointCloud<pcl::PointXYZI>::Ptr pose_cloud(
      new pcl::PointCloud<pcl::PointXYZI>);
  bool is_build_descriptor = false;
  int key_frame_id = 0;
  int count = 0;
  for (size_t i = 0; i < frame_number_list.size(); i++) {
    if (count == 0) {
      pcl::PointCloud<pcl::PointXYZI>::Ptr temp_cloud(
          new pcl::PointCloud<pcl::PointXYZI>);
      key_cloud_list.push_back(temp_cloud);
    }
    std::stringstream single_lidar_path;
    single_lidar_path << lidar_data_path << "/velodyne_points/data/"
                      << std::setfill('0') << std::setw(10)
                      << frame_number_list[i] << ".bin";
    std::vector<float> lidar_data = read_lidar_data(single_lidar_path.str());
    Eigen::Vector3d translation = pose_list[i].first;
    Eigen::Matrix3d rotation = pose_list[i].second;
    std::vector<Eigen::Vector3d> lidar_points;
    std::vector<float> lidar_intensities;
    Eigen::Quaterniond q(rotation);
    nav_msgs::Odometry odom;
    odom.header.frame_id = "camera_init";
    odom.pose.pose.position.x = translation[0];
    odom.pose.pose.position.y = translation[1];
    odom.pose.pose.position.z = translation[2];
    odom.pose.pose.orientation.w = q.w();
    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();
    pubOdomAftMapped.publish(odom);
    loop.sleep();
    pcl::PointCloud<pcl::PointXYZI>::Ptr register_cloud(
        new pcl::PointCloud<pcl::PointXYZI>);
    for (std::size_t j = 0; j < lidar_data.size(); j += 4) {
      lidar_points.emplace_back(lidar_data[j], lidar_data[j + 1],
                                lidar_data[j + 2]);
      lidar_intensities.push_back(lidar_data[j + 3]);
      Eigen::Vector3d pi(lidar_data[j], lidar_data[j + 1], lidar_data[j + 2]);
      pi = lidar_to_pose_rot * pi + lidar_to_pose_t;
      pi = rotation * pi + translation;
      pcl::PointXYZI point;
      point.x = pi[0];
      point.y = pi[1];
      point.z = pi[2];
      point.intensity = lidar_data[j + 3];
      register_cloud->push_back(point);
    }
    down_sampling_voxel(*register_cloud, 0.5);
    // std::cout << "here" << std::endl;
    for (size_t i = 0; i < register_cloud->size(); i++) {
      key_cloud_list.back()->points.push_back(register_cloud->points[i]);
    }
    if (count == config_setting.sub_frame_num_ / 2) {
      current_mean_position = translation;
      pcl::PointXYZI pi;
      pi.x = current_mean_position[0];
      pi.y = current_mean_position[1];
      pi.z = current_mean_position[2];
      pi.intensity = pose_cnt;
      pose_cnt++;
      pose_cloud->points.push_back(pi);
    }
    if (count < config_setting.sub_frame_num_ - 1) {
      count++;
    } else {
      count = 0;
      is_build_descriptor = true;
    }
    if (is_build_descriptor) {
      is_build_descriptor = false;
      down_sampling_voxel(*key_cloud_list[key_frame_id], 0.5);
      std::cout << "key frame:" << key_frame_id
                << ", cloud size:" << key_cloud_list[key_frame_id]->size()
                << std::endl;
      key_frame_id++;
    }
  }

  pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kd_tree(
      new pcl::KdTreeFLANN<pcl::PointXYZI>);
  kd_tree->setInputCloud(pose_cloud);
  std::vector<int> indices;
  std::vector<float> distances;
  double radius = 50;
  double overlap_threshold = 0.5;
  int gt_loop_num = 0;
  for (int i = 0; i < pose_cloud->size(); i++) {
    double max_overlap = 0;
    bool trigger_loop = false;
    int loop_id = 0;
    pcl::PointXYZI searchPoint = pose_cloud->points[i];
    int size = kd_tree->radiusSearch(searchPoint, radius, indices, distances);
    for (int j = 0; j < size; j++) {
      if (indices[j] >= i - 20) {
        continue;
      } else {
        pcl::PointCloud<pcl::PointXYZI> ds_cloud1 = *key_cloud_list[i];
        pcl::PointCloud<pcl::PointXYZI> ds_cloud2 = *key_cloud_list[indices[j]];
        // down_sampling_voxel(ds_cloud1, 0.5);
        // down_sampling_voxel(ds_cloud2, 0.5);
        double overlap =
            calc_overlap(ds_cloud1.makeShared(), ds_cloud2.makeShared(), 0.5);
        if (overlap > max_overlap) {
          max_overlap = overlap;
          loop_id = indices[j];
        }
      }
    }
    if (max_overlap > overlap_threshold) {
      trigger_loop = true;
      gt_loop_num++;
      sensor_msgs::PointCloud2 pub_cloud;
      pcl::toROSMsg(*key_cloud_list[i], pub_cloud);
      pub_cloud.header.frame_id = "camera_init";
      pubCureentCloud.publish(pub_cloud);
      loop.sleep();
      pcl::toROSMsg(*key_cloud_list[loop_id], pub_cloud);
      pub_cloud.header.frame_id = "camera_init";
      pubMatchedCloud.publish(pub_cloud);
      loop.sleep();
      result_file << i << "," << searchPoint.x << "," << searchPoint.y << ","
                  << searchPoint.z << "," << 1 << "," << loop_id << ","
                  << max_overlap << std::endl;
      max_overlap = floor((max_overlap * pow(10, 3) + 0.5)) / pow(10, 3);
      std::cout << "loop trigger:" << i << "-" << loop_id
                << ", overlap:" << max_overlap << std::endl;
      std::string max_overlap_str = std::to_string(max_overlap);
      max_overlap_str =
          max_overlap_str.substr(0, max_overlap_str.find(".") + 4);
      max_overlap_str = "Overlap: " + max_overlap_str;
      // publish_map(pubLaserCloudMap);
      cv::Mat max_overlap_pic = cv::Mat::zeros(200, 800, CV_8UC3);
      cv::Point siteNO;
      siteNO.x = 100;
      siteNO.y = 100;
      cv::putText(max_overlap_pic, max_overlap_str, siteNO, 4, 2,
                  cv::Scalar(255, 255, 255), 4);
      cv::imshow("", max_overlap_pic);
      cv::waitKey(500);
      // getchar();
    } else {
      result_file << i << "," << searchPoint.x << "," << searchPoint.y << ","
                  << searchPoint.z << "," << 0 << "," << 0 << "," << 0
                  << std::endl;
    }
  }
}