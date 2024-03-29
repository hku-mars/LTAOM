#include "include/std.h"
#include "include/std_ba.h"
#include <nav_msgs/Odometry.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud2.h>

#define debug

int findPoseIndexUsingTime(std::vector<double> &time_list, long &time) {
  long time_inc = 10000000000;
  int min_index = -1;
  for (size_t i = 0; i < time_list.size(); i++) {
    if (fabs(time_list[i] - time) < time_inc) {
      time_inc = fabs(time_list[i] - time);
      min_index = i;
    }
  }
  return min_index;
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "std_loop");
  ros::NodeHandle nh;
  std::string data_name = "";
  std::string setting_path = "";
  std::string bag_file1 = "";
  std::string bag_file2 = "";
  std::string pose_file1 = "";
  std::string pose_file2 = "";
  std::string loop_gt_file = "";
  double icp_threshold = 0.5;
  bool calc_gt_enable = false;
  nh.param<std::string>("data_name", data_name, "");
  nh.param<std::string>("setting_path", setting_path, "");
  nh.param<std::string>("bag_file1", bag_file1, "");
  nh.param<std::string>("pose_file1", pose_file1, "");
  nh.param<std::string>("bag_file2", bag_file2, "");
  nh.param<std::string>("pose_file2", pose_file2, "");
  nh.param<std::string>("loop_gt_file", loop_gt_file, "");
  nh.param<bool>("calc_gt_enable", calc_gt_enable, false);

  nh.param<double>("icp_threshold", icp_threshold, 0.5);
  std::string icp_string = std::to_string(icp_threshold);
  std::string result_path =
      "/home/ycj/matlab_code/loop_detection/result/" + data_name + "/" +
      data_name + "_" + icp_string.substr(0, icp_string.find(".") + 3) + ".txt";
  std::ofstream result_file(result_path);
  std::ofstream debug_file("/home/ycj/catkin_ws/src/STD/Log/log.txt");

  ros::Publisher pubOdomAftMapped =
      nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 10);
  ros::Publisher pubRegisterCloud =
      nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100);
  ros::Publisher pubFirstCloud =
      nh.advertise<sensor_msgs::PointCloud2>("/cloud_first", 100);

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
  ros::Publisher pubCorrectCloud =
      nh.advertise<sensor_msgs::PointCloud2>("/cloud_correct", 10000);
  ros::Publisher pubOdomCorreted =
      nh.advertise<nav_msgs::Odometry>("/odom_corrected", 10);

  ros::Rate loop(50000);
  ros::Rate late_loop(100);

  ConfigSetting config_setting;
  load_config_setting(setting_path, config_setting);

  // load first bag
  std::vector<std::pair<Eigen::Vector3d, Eigen::Matrix3d>> pose_list1;
  std::vector<double> time_list1;
  if (config_setting.gt_file_style_ == 0) {
    load_pose_with_time(pose_file1, pose_list1, time_list1);
  } else if (config_setting.gt_file_style_ == 1) {
    load_cu_pose_with_time(pose_file1, pose_list1, time_list1);
  }

  std::string print_msg = "Sucessfully load pose file:" + pose_file1 +
                          ". pose size:" + std::to_string(time_list1.size());
  ROS_INFO_STREAM(print_msg.c_str());

  // save all point clouds of key frame
  std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> key_cloud_list;

  // save all point clouds of key frame
  std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> key_cloud_list2;

  // save all planes of key frame
  std::vector<pcl::PointCloud<pcl::PointXYZINormal>::Ptr> history_plane_list;

  // save all binary descriptors of key frame
  std::vector<std::vector<BinaryDescriptor>> history_binary_list;

  // save all STD descriptors of key frame
  std::vector<std::vector<STD>> history_STD_list;

  // save all poses(translation, rotation) of all frame
  std::vector<std::pair<Eigen::Vector3d, Eigen::Matrix3d>> normal_pose_list;

  // save all optimized poses
  std::vector<double *> opt_pose_list;

  // save all history poses
  std::vector<double *> history_pose_list;

  // hash table, save all descriptor
  std::unordered_map<STD_LOC, std::vector<STD>> STD_map;

  std::vector<PlanePair> plane_match_list;
  // calc mean time
  double mean_time = 0;

  // record mean position
  Eigen::Vector3d current_mean_position(0, 0, 0);

  long current_time = 0;

  // load lidar point cloud and start loop
  pcl::PointCloud<pcl::PointXYZI>::Ptr pose_cloud(
      new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr current_key_cloud(
      new pcl::PointCloud<pcl::PointXYZI>);
  bool is_build_descriptor = false;
  int key_frame_id = 0;
  int key_frame_last = 0;
  std::fstream file1_;
  file1_.open(bag_file1, std::ios::in);
  if (!file1_) {
    std::cout << "File " << bag_file1 << " does not exit" << std::endl;
  }
  ROS_INFO("Start to load the rosbag %s", bag_file1.c_str());
  rosbag::Bag bag1;
  try {
    bag1.open(bag_file1, rosbag::bagmode::Read);
  } catch (rosbag::BagException e) {
    ROS_ERROR_STREAM("LOADING BAG FAILED: " << e.what());
  }
  std::vector<std::string> types;
  types.push_back(std::string("sensor_msgs/PointCloud2"));
  rosbag::View view1(bag1, rosbag::TypeQuery(types));
  bool is_init_bag = false;
  bool is_skip_frame = false;
  Eigen::Vector3d init_translation;
  Eigen::Vector3d last_translation;
  Eigen::Vector3d current_translation;
  Eigen::Matrix3d current_rotation;
  Eigen::Quaterniond last_q;
  int count = 0;
  auto t_load_map_begin = std::chrono::high_resolution_clock::now();
  BOOST_FOREACH (rosbag::MessageInstance const m, view1) {
    sensor_msgs::PointCloud2::ConstPtr cloud_ptr =
        m.instantiate<sensor_msgs::PointCloud2>();
    if (cloud_ptr != NULL) {
      if (count == 0) {
        if (!is_skip_frame) {

          current_key_cloud->clear();
        }
      }
      long laser_time = cloud_ptr->header.stamp.toNSec();
      pcl::PCLPointCloud2 pcl_pc;
      pcl::PointCloud<pcl::PointXYZI> pcl_cloud;
      pcl_conversions::toPCL(*cloud_ptr, pcl_pc);
      pcl::fromPCLPointCloud2(pcl_pc, pcl_cloud);
      int pose_index = findPoseIndexUsingTime(time_list1, laser_time);
      if (pose_index < 0) {
        is_skip_frame = true;
        continue;
      }
      is_skip_frame = false;
      Eigen::Vector3d translation = pose_list1[pose_index].first;
      Eigen::Matrix3d rotation = pose_list1[pose_index].second;
      Eigen::Quaterniond q(rotation);

      if (!is_init_bag) {
        init_translation = translation;
        translation << 0, 0, 0;
        is_init_bag = true;
        last_translation = translation;
        last_q = q;
      } else {
        translation = translation - init_translation;
      }
      if (config_setting.stop_skip_enable_) {
        Eigen::Vector3d position_inc;
        position_inc = translation - last_translation;
        double rotation_inc = q.angularDistance(last_q);
        if (position_inc.norm() < 0.2 && rotation_inc < DEG2RAD(5)) {
          continue;
        }
        last_translation = translation;
        last_q = q;
      }

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
      for (size_t i = 0; i < pcl_cloud.size(); i++) {
        Eigen::Vector3d pv(pcl_cloud.points[i].x, pcl_cloud.points[i].y,
                           pcl_cloud.points[i].z);
        pv = config_setting.rot_lidar_to_vehicle_ * pv +
             config_setting.t_lidar_to_vehicle_;
        pv = rotation * pv + translation;
        pcl::PointXYZI pi = pcl_cloud.points[i];
        pi.x = pv[0];
        pi.y = pv[1];
        pi.z = pv[2];
        register_cloud->push_back(pi);
      }
      if (count == 0) {
        if (!is_skip_frame) {
          current_translation = translation;
          current_rotation = rotation;
        }
      }

      time_t ds_start = clock();
      down_sampling_voxel(*register_cloud, config_setting.ds_size_);
      for (size_t i = 0; i < register_cloud->size(); i++) {
        current_key_cloud->points.push_back(register_cloud->points[i]);
      }
      if (count == config_setting.sub_frame_num_ / 2) {
        current_time = cloud_ptr->header.stamp.toNSec() / 1000;
        current_mean_position = translation;
        pcl::PointXYZI pi;
        pi.x = current_mean_position[0];
        pi.y = current_mean_position[1];
        pi.z = current_mean_position[2];
        pose_cloud->points.push_back(pi);
      }
      if (count < config_setting.sub_frame_num_ - 1) {
        count++;
      } else {
        count = 0;
        is_build_descriptor = true;
      }
    }
    if (is_build_descriptor) {
      Eigen::Quaterniond quaternion(current_rotation);
      double *pose = new double[7];
      pose[0] = current_translation[0];
      pose[1] = current_translation[1];
      pose[2] = current_translation[2];
      pose[3] = quaternion.x();
      pose[4] = quaternion.y();
      pose[5] = quaternion.z();
      pose[6] = quaternion.w();
      opt_pose_list.push_back(pose);
      std::cout << std::endl;
      std::cout << "Key Frame:" << key_frame_id
                << ", cloud size:" << current_key_cloud->size() << std::endl;
      debug_file << std::endl;
      debug_file << "Key frame:" << key_frame_id
                 << ", cloud size:" << current_key_cloud->size() << std::endl;

      auto t1 = std::chrono::high_resolution_clock::now();
      auto t_build_descriptor_begin = std::chrono::high_resolution_clock::now();
      std::unordered_map<VOXEL_LOC, OctoTree *> voxel_map;
      init_voxel_map(config_setting, *current_key_cloud, voxel_map);
      pcl::PointCloud<pcl::PointXYZINormal>::Ptr frame_plane_cloud(
          new pcl::PointCloud<pcl::PointXYZINormal>);
      get_plane(voxel_map, frame_plane_cloud);
      history_plane_list.push_back(frame_plane_cloud);
      std::vector<Plane *> proj_plane_list;
      std::vector<Plane *> merge_plane_list;
      get_project_plane(config_setting, voxel_map, proj_plane_list);
      sort(proj_plane_list.begin(), proj_plane_list.end(), plane_greater_sort);
      merge_plane(config_setting, proj_plane_list, merge_plane_list);
      sort(merge_plane_list.begin(), merge_plane_list.end(),
           plane_greater_sort);
      std::vector<BinaryDescriptor> binary_list;
      std::vector<BinaryDescriptor> binary_around_list;
      binary_extractor(config_setting, merge_plane_list, current_key_cloud,
                       binary_list);
      history_binary_list.push_back(binary_list);
      std::vector<STD> STD_list;
      generate_std(config_setting, binary_list, key_frame_id, STD_list);
      auto t_add_descriptor_begin = std::chrono::high_resolution_clock::now();
      add_STD(STD_map, STD_list);
      auto t_add_descriptor_end = std::chrono::high_resolution_clock::now();
      // down_sampling_voxel(*current_key_cloud, 0.5);
      is_build_descriptor = false;
      pcl::PointCloud<pcl::PointXYZI>::Ptr temp_cloud(
          new pcl::PointCloud<pcl::PointXYZI>);
      key_cloud_list.push_back(temp_cloud);
      for (size_t i = 0; i < current_key_cloud->size(); i++) {
        key_cloud_list.back()->push_back(current_key_cloud->points[i]);
      }
      if (config_setting.is_kitti_) {
        result_file << key_frame_id << "," << current_mean_position[0] << ","
                    << current_mean_position[1] << ","
                    << current_mean_position[2] << ",";
      } else {
        result_file << key_frame_id << "," << current_time << ","
                    << current_mean_position[0] << ","
                    << current_mean_position[1] << ","
                    << current_mean_position[2] << ",";
      }
      result_file << 0 << "," << 0 << "," << 0 << "," << 0 << "," << 0 << ","
                  << 0 << "," << 0 << "," << 0 << "," << 0 << "," << 0 << ","
                  << 0 << "," << 0 << "," << 0 << "," << 0 << "," << 0
                  << std::endl;
      key_frame_id++;
      key_frame_last++;
      for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++) {
        delete (iter->second);
      }
    }
  }

  auto t_load_map_end = std::chrono::high_resolution_clock::now();
  std::cout << "load first bag finish! Time cost:"
            << time_inc(t_load_map_end, t_load_map_begin) / 1000.0 << " s"
            << std::endl;
  PoseOptimizer pose_optimizer(config_setting);
  // load second bag
  is_skip_frame = false;
  count = 0;
  std::vector<std::pair<Eigen::Vector3d, Eigen::Matrix3d>> pose_list2;
  std::vector<double> time_list2;
  if (config_setting.gt_file_style_ == 0) {
    load_pose_with_time(pose_file2, pose_list2, time_list2);
  } else if (config_setting.gt_file_style_ == 1) {
    load_cu_pose_with_time(pose_file2, pose_list2, time_list2);
  }

  std::string print_msg2 = "Sucessfully load pose file:" + pose_file2 +
                           ". pose size:" + std::to_string(time_list2.size());
  ROS_INFO_STREAM(print_msg2.c_str());
  std::fstream file2_;
  file2_.open(bag_file2, std::ios::in);
  if (!file2_) {
    std::cout << "File " << bag_file2 << " does not exit" << std::endl;
  }
  ROS_INFO("Start to load the rosbag %s", bag_file2.c_str());
  rosbag::Bag bag2;
  try {
    bag2.open(bag_file2, rosbag::bagmode::Read);
  } catch (rosbag::BagException e) {
    ROS_ERROR_STREAM("LOADING BAG FAILED: " << e.what());
  }
  rosbag::View view2(bag2, rosbag::TypeQuery(types));
  BOOST_FOREACH (rosbag::MessageInstance const m, view2) {
    sensor_msgs::PointCloud2::ConstPtr cloud_ptr =
        m.instantiate<sensor_msgs::PointCloud2>();
    if (cloud_ptr != NULL) {
      if (count == 0) {
        if (!is_skip_frame) {
          current_key_cloud->clear();
        }
      }
      long laser_time = cloud_ptr->header.stamp.toNSec();
      pcl::PCLPointCloud2 pcl_pc;
      pcl::PointCloud<pcl::PointXYZI> pcl_cloud;
      pcl_conversions::toPCL(*cloud_ptr, pcl_pc);
      pcl::fromPCLPointCloud2(pcl_pc, pcl_cloud);
      int pose_index = findPoseIndexUsingTime(time_list2, laser_time);
      if (pose_index < 0) {
        is_skip_frame = true;
        continue;
      }
      is_skip_frame = false;
      Eigen::Vector3d translation = pose_list2[pose_index].first;
      Eigen::Matrix3d rotation = pose_list2[pose_index].second;
      Eigen::Quaterniond q(rotation);

      // if (!is_init_bag) {
      //   init_translation = translation;
      //   translation << 0, 0, 0;
      //   is_init_bag = true;
      //   last_translation = translation;
      //   last_q = q;
      // } else {

      // }
      translation = translation - init_translation;
      if (config_setting.stop_skip_enable_) {
        Eigen::Vector3d position_inc;
        position_inc = translation - last_translation;
        double rotation_inc = q.angularDistance(last_q);
        if (position_inc.norm() < 0.2 && rotation_inc < DEG2RAD(5)) {
          continue;
        }
        last_translation = translation;
        last_q = q;
      }

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
      for (size_t i = 0; i < pcl_cloud.size(); i++) {
        Eigen::Vector3d pv(pcl_cloud.points[i].x, pcl_cloud.points[i].y,
                           pcl_cloud.points[i].z);
        pv = config_setting.rot_lidar_to_vehicle_ * pv +
             config_setting.t_lidar_to_vehicle_;
        pv = rotation * pv + translation;
        pcl::PointXYZI pi = pcl_cloud.points[i];
        pi.x = pv[0];
        pi.y = pv[1];
        pi.z = pv[2];
        register_cloud->push_back(pi);
      }

      std::pair<Eigen::Vector3d, Eigen::Matrix3d> single_pose;
      single_pose.first = translation;
      single_pose.second = rotation;
      normal_pose_list.push_back(single_pose);
      Eigen::Quaterniond quaternion(rotation);
      if (count == 0) {
        if (!is_skip_frame) {
          current_translation = translation;
          current_rotation = rotation;
        }
      }

      time_t ds_start = clock();
      down_sampling_voxel(*register_cloud, config_setting.ds_size_);
      for (size_t i = 0; i < register_cloud->size(); i++) {
        current_key_cloud->points.push_back(register_cloud->points[i]);
      }
      if (count == config_setting.sub_frame_num_ / 2) {
        current_time = cloud_ptr->header.stamp.toNSec() / 1000;
        current_mean_position = translation;
        pcl::PointXYZI pi;
        pi.x = current_mean_position[0];
        pi.y = current_mean_position[1];
        pi.z = current_mean_position[2];
        pose_cloud->points.push_back(pi);
      }
      if (count < config_setting.sub_frame_num_ - 1) {
        count++;
      } else {
        count = 0;
        is_build_descriptor = true;
      }
    }
    if (is_build_descriptor) {
      Eigen::Quaterniond quaternion(current_rotation);
      double *pose = new double[7];
      pose[0] = current_translation[0];
      pose[1] = current_translation[1];
      pose[2] = current_translation[2];
      pose[3] = quaternion.x();
      pose[4] = quaternion.y();
      pose[5] = quaternion.z();
      pose[6] = quaternion.w();
      opt_pose_list.push_back(pose);
      if (config_setting.is_kitti_) {
        result_file << key_frame_id << "," << current_mean_position[0] << ","
                    << current_mean_position[1] << ","
                    << current_mean_position[2] << ",";
      } else {
        result_file << key_frame_id << "," << current_time << ","
                    << current_mean_position[0] << ","
                    << current_mean_position[1] << ","
                    << current_mean_position[2] << ",";
      }
      std::cout << std::endl;
      std::cout << "Key Frame:" << key_frame_id
                << ", cloud size:" << current_key_cloud->size() << std::endl;
      debug_file << std::endl;
      debug_file << "Key frame:" << key_frame_id
                 << ", cloud size:" << current_key_cloud->size() << std::endl;

      auto t1 = std::chrono::high_resolution_clock::now();
      auto t_build_descriptor_begin = std::chrono::high_resolution_clock::now();
      std::unordered_map<VOXEL_LOC, OctoTree *> voxel_map;
      init_voxel_map(config_setting, *current_key_cloud, voxel_map);
      pcl::PointCloud<pcl::PointXYZINormal>::Ptr frame_plane_cloud(
          new pcl::PointCloud<pcl::PointXYZINormal>);
      get_plane(voxel_map, frame_plane_cloud);
      history_plane_list.push_back(frame_plane_cloud);
      std::vector<Plane *> proj_plane_list;
      std::vector<Plane *> merge_plane_list;
      get_project_plane(config_setting, voxel_map, proj_plane_list);
      sort(proj_plane_list.begin(), proj_plane_list.end(), plane_greater_sort);
      merge_plane(config_setting, proj_plane_list, merge_plane_list);
      sort(merge_plane_list.begin(), merge_plane_list.end(),
           plane_greater_sort);

      std::vector<BinaryDescriptor> binary_list;
      std::vector<BinaryDescriptor> binary_around_list;
      binary_extractor(config_setting, merge_plane_list, current_key_cloud,
                       binary_list);

      std::vector<STD> STD_list;
      generate_std(config_setting, binary_list, key_frame_id, STD_list);
      auto t_build_descriptor_end = std::chrono::high_resolution_clock::now();
      sensor_msgs::PointCloud2 pub_cloud;
      pcl::toROSMsg(*current_key_cloud, pub_cloud);
      pub_cloud.header.frame_id = "camera_init";
      pubCureentCloud.publish(pub_cloud);
      loop.sleep();
      pcl::PointCloud<pcl::PointXYZ> key_points_cloud;
      for (auto var : binary_list) {
        pcl::PointXYZ pi;
        pi.x = var.location_[0];
        pi.y = var.location_[1];
        pi.z = var.location_[2];
        key_points_cloud.push_back(pi);
      }
      pcl::toROSMsg(key_points_cloud, pub_cloud);
      pub_cloud.header.frame_id = "camera_init";
      pubCurrentBinary.publish(pub_cloud);
      loop.sleep();
      Eigen::Vector3d color1(1, 0, 0);
      publish_binary(binary_list, color1, "current", pubSTD);
      // candidate search
      auto t_candidate_search_begin = std::chrono::high_resolution_clock::now();
      std::vector<STDMatchList> alternative_match;
      candidate_searcher_old(config_setting, STD_map, STD_list,
                             alternative_match);
      auto t_candidate_search_end = std::chrono::high_resolution_clock::now();

      // geometrical verification
      auto t_fine_loop_begin = std::chrono::high_resolution_clock::now();
      bool triggle_loop = false;
      Eigen::Vector3d best_t;
      Eigen::Matrix3d best_rot;
      Eigen::Vector3d loop_translation;
      Eigen::Matrix3d loop_rotation;
      std::vector<std::pair<STD, STD>> sucess_match_list;
      std::vector<std::pair<STD, STD>> unsucess_match_list;
      std::vector<std::pair<STD, STD>> sucess_match_list_publish;
      std::vector<std::pair<STD, STD>> unsucess_match_list_publish;
      int match_size = 0;
      int rough_size = 0;
      int candidate_id = -1;
      double mean_triangle_dis = 0;
      double mean_binary_similarity = 0;
      double outlier_mean_triangle_dis = 0;
      double outlier_mean_binary_similarity = 0;
      int match_frame = 0;
      double best_score = 0;
      double best_icp_score = 0;
      int best_frame = -1;
      for (int i = 0; i < alternative_match.size(); i++) {
        if (alternative_match[i].match_list_.size() >= 4) {
          bool fine_sucess = false;
          Eigen::Matrix3d std_rot;
          Eigen::Vector3d std_t;
#ifdef debug
          debug_file << "[Rough match] rough match frame:"
                     << alternative_match[i].match_frame_ << " match size:"
                     << alternative_match[i].match_list_.size() << std::endl;
#endif
          sucess_match_list.clear();
          fine_loop_detection_tbb(
              config_setting, alternative_match[i].match_list_, fine_sucess,
              std_rot, std_t, sucess_match_list, unsucess_match_list);
          if (fine_sucess) {
            double icp_score = geometric_verify(
                config_setting, frame_plane_cloud,
                history_plane_list[alternative_match[i].match_frame_], std_rot,
                std_t);
            double score = icp_score + sucess_match_list.size() * 1.0 / 1000;
#ifdef debug
            debug_file << "Fine sucess, Fine size:" << sucess_match_list.size()
                       << "  ,Icp score:" << icp_score << ", score:" << score
                       << std::endl;
#endif
            if (score > best_score) {
              unsucess_match_list_publish = unsucess_match_list;
              sucess_match_list_publish = sucess_match_list;
              best_frame = alternative_match[i].match_frame_;
              best_score = score;
              best_icp_score = icp_score;
              best_rot = std_rot;
              best_t = std_t;
              rough_size = alternative_match[i].match_list_.size();
              match_size = sucess_match_list.size();
              candidate_id = i;
            }
          }
        }
      }
      if (best_icp_score > icp_threshold) {
        loop_translation = best_t;
        loop_rotation = best_rot;
        match_frame = best_frame;
        triggle_loop = true;
        mean_triangle_dis = calc_triangle_dis(sucess_match_list_publish);
        mean_binary_similarity =
            calc_binary_similaity(sucess_match_list_publish);
        outlier_mean_triangle_dis =
            calc_triangle_dis(unsucess_match_list_publish);
        outlier_mean_triangle_dis =
            calc_binary_similaity(unsucess_match_list_publish);
      } else {
        triggle_loop = false;
      }
      auto t_fine_loop_end = std::chrono::high_resolution_clock::now();
      is_build_descriptor = false;
      if (triggle_loop) {
#ifdef debug
        debug_file << "[Loop Sucess] " << key_frame_id << "--" << match_frame
                   << ", candidate id:" << candidate_id
                   << ", icp:" << best_score << std::endl;
        debug_file << "[Loop Info] "
                   << "rough size:" << rough_size
                   << ", match size:" << match_size
                   << ", rough triangle dis:" << outlier_mean_triangle_dis
                   << ", fine triangle dis:" << mean_triangle_dis
                   << ", rough binary similarity:" << outlier_mean_triangle_dis
                   << ", fine binary similarity:" << mean_binary_similarity
                   << std::endl;
#endif
        result_file << 1 << "," << match_frame << "," << candidate_id << ","
                    << match_size << "," << rough_size << ","
                    << loop_translation[0] << "," << loop_translation[1] << ","
                    << loop_translation[2] << ",";
        result_file << 1 << "," << 1 << "," << best_score << ","
                    << time_inc(t_build_descriptor_end,
                                t_build_descriptor_begin)
                    << ","
                    << time_inc(t_candidate_search_end,
                                t_candidate_search_begin)
                    << "," << time_inc(t_fine_loop_end, t_fine_loop_begin)
                    << "," << 0 << std::endl;
        pcl::toROSMsg(*key_cloud_list[match_frame], pub_cloud);
        pub_cloud.header.frame_id = "camera_init";
        pubMatchedCloud.publish(pub_cloud);
        loop.sleep();
        pcl::PointCloud<pcl::PointXYZ> matched_key_points_cloud;
        for (auto var : history_binary_list[match_frame]) {
          pcl::PointXYZ pi;
          pi.x = var.location_[0];
          pi.y = var.location_[1];
          pi.z = var.location_[2];
          matched_key_points_cloud.push_back(pi);
        }
        pcl::toROSMsg(matched_key_points_cloud, pub_cloud);
        pub_cloud.header.frame_id = "camera_init";
        pubMatchedBinary.publish(pub_cloud);
        Eigen::Vector3d color2(0, 1, 0);
        publish_binary(history_binary_list[match_frame], color2, "history",
                       pubSTD);
        loop.sleep();
        publish_std(sucess_match_list_publish, pubSTD);

        geometric_icp(frame_plane_cloud, history_plane_list[match_frame],
                      loop_rotation, loop_translation);
        std::pair<Eigen::Vector3d, Eigen::Matrix3d> initial_guess;
        initial_guess.first = loop_translation;
        initial_guess.second = loop_rotation;
        std::cout << "loop translation: " << loop_translation.transpose()
                  << std::endl
                  << ", loop rotation:" << loop_rotation << std::endl;
        // getchar();
        std::vector<PlanePair> loop_plane_pair;
        pose_optimizer.addConnection(
            opt_pose_list.back(), opt_pose_list[match_frame], initial_guess,
            frame_plane_cloud, history_plane_list[match_frame],
            loop_plane_pair);
        pose_optimizer.problem_.SetParameterBlockConstant(
            opt_pose_list[match_frame]);
        for (auto var : loop_plane_pair) {
          var.source_id = key_frame_id;
          var.target_id = match_frame;
          plane_match_list.push_back(var);
        }
        // add near connection
        if (opt_pose_list.size() >= (key_frame_last + 1)) {
          initial_guess.first << 0, 0, 0;
          initial_guess.second = Eigen::Matrix3d::Identity();
          std::vector<PlanePair> near_plane_pair;
          pose_optimizer.addConnection(
              opt_pose_list.back(), opt_pose_list.at(opt_pose_list.size() - 2),
              initial_guess, frame_plane_cloud,
              history_plane_list.at(history_plane_list.size() - 2),
              near_plane_pair);
          for (auto var : near_plane_pair) {
            var.source_id = key_frame_id;
            var.target_id = key_frame_id - 1;
            plane_match_list.push_back(var);
          }
        }

      } else {
        debug_file << "[Loop Fail] " << key_frame_id << ", icp:" << best_score
                   << std::endl;
        result_file
            << 0 << "," << 0 << "," << 0 << "," << 0 << "," << 0 << "," << 0
            << "," << 0 << "," << 0 << "," << 0 << "," << 0 << "," << 0 << ","
            << time_inc(t_build_descriptor_end, t_build_descriptor_begin) << ","
            << time_inc(t_candidate_search_end, t_candidate_search_begin) << ","
            << time_inc(t_fine_loop_end, t_fine_loop_begin) << "," << 0
            << std::endl;
        // add near connection
        if (opt_pose_list.size() >= (key_frame_last + 1)) {
          std::pair<Eigen::Vector3d, Eigen::Matrix3d> initial_guess;
          initial_guess.first << 0, 0, 0;
          initial_guess.second = Eigen::Matrix3d::Identity();
          std::vector<PlanePair> near_plane_pair;
          pose_optimizer.addConnection(
              opt_pose_list.back(), opt_pose_list.at(opt_pose_list.size() - 2),
              initial_guess, frame_plane_cloud,
              history_plane_list.at(history_plane_list.size() - 2),
              near_plane_pair);
          for (auto var : near_plane_pair) {
            var.source_id = key_frame_id;
            var.target_id = key_frame_id - 1;
            plane_match_list.push_back(var);
          }
        }
      }
      pcl::PointCloud<pcl::PointXYZI>::Ptr temp_cloud(
          new pcl::PointCloud<pcl::PointXYZI>);
      key_cloud_list2.push_back(temp_cloud);
      // down_sampling_voxel(*current_key_cloud, 0.5);
      for (size_t i = 0; i < current_key_cloud->size(); i++) {
        key_cloud_list2.back()->push_back(current_key_cloud->points[i]);
      }
      key_frame_id++;

      for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++) {
        delete (iter->second);
      }
    }
  }
  // return 0;
  bool ba_enable = false;

  std::vector<std::pair<Eigen::Vector3d, Eigen::Matrix3d>> origin_pose_list;
  for (size_t i = 0; i < opt_pose_list.size(); i++) {
    std::pair<Eigen::Vector3d, Eigen::Matrix3d> single_pose;
    Eigen::Vector3d translation(opt_pose_list[i][0], opt_pose_list[i][1],
                                opt_pose_list[i][2]);
    Eigen::Quaterniond quaternion(opt_pose_list[i][6], opt_pose_list[i][3],
                                  opt_pose_list[i][4], opt_pose_list[i][5]);
    single_pose.first = translation;
    single_pose.second = quaternion.toRotationMatrix();
    origin_pose_list.push_back(single_pose);
  }

  // calc residual before opt
  bool calc_residual_en = true;
  if (calc_residual_en) {
    double near_dis_residual = 0;
    double near_normal_residual = 0;
    int near_cnt = 0;
    double loop_dis_residual = 0;
    double loop_normal_residual = 0;
    int loop_cnt = 0;
    for (size_t i = 0; i < plane_match_list.size(); i++) {
      PlanePair pp = plane_match_list[i];
      Eigen::Vector3d source_normal = normal2vec(pp.source_plane);
      Eigen::Vector3d source_centriod = point2vec(pp.source_plane);
      source_normal = origin_pose_list[pp.source_id].second * source_normal;
      source_centriod =
          origin_pose_list[pp.source_id].second * source_centriod +
          origin_pose_list[pp.source_id].first;

      Eigen::Vector3d target_normal = normal2vec(pp.target_plane);
      Eigen::Vector3d target_centriod = point2vec(pp.target_plane);

      target_normal = origin_pose_list[pp.target_id].second * target_normal;
      target_centriod =
          origin_pose_list[pp.target_id].second * target_centriod +
          origin_pose_list[pp.target_id].first;

      if ((pp.source_id - pp.target_id) == 1) {
        double dis = fabs(target_normal.transpose() *
                          (source_centriod - target_centriod));
        double normal_diff = std::min((source_normal + target_normal).norm(),
                                      (source_normal - target_normal).norm());
        near_dis_residual += dis;
        near_normal_residual += normal_diff;
        near_cnt++;
      } else {
        double dis = fabs(target_normal.transpose() *
                          (source_centriod - target_centriod));
        double normal_diff = std::min((source_normal + target_normal).norm(),
                                      (source_normal - target_normal).norm());
        loop_dis_residual += dis;
        loop_normal_residual += normal_diff;
        loop_cnt++;
      }
    }
    std::cout << "residual before opt, near dis: "
              << near_dis_residual / near_cnt
              << ", near normal: " << near_normal_residual / near_cnt
              << std::endl;
    std::cout << "residual before opt, loop dis: "
              << loop_dis_residual / loop_cnt
              << ", near normal: " << loop_normal_residual / loop_cnt
              << std::endl;
  }
  auto t_ba_begin = std::chrono::high_resolution_clock::now();
  std::cout << "Solving ba problem!" << std::endl;
  pose_optimizer.Solve();
  auto t_ba_end = std::chrono::high_resolution_clock::now();
  std::cout << "Final ba time cost: " << time_inc(t_ba_end, t_ba_begin) / 1000.0
            << " s" << std::endl;
  if (calc_residual_en) {
    double near_dis_residual = 0;
    double near_normal_residual = 0;
    int near_cnt = 0;
    double loop_dis_residual = 0;
    double loop_normal_residual = 0;
    int loop_cnt = 0;
    for (size_t i = 0; i < plane_match_list.size(); i++) {
      PlanePair pp = plane_match_list[i];
      Eigen::Vector3d source_normal(pp.source_plane.normal_x,
                                    pp.source_plane.normal_y,
                                    pp.source_plane.normal_z);
      Eigen::Vector3d source_centriod(pp.source_plane.x, pp.source_plane.y,
                                      pp.source_plane.z);
      Eigen::Vector3d opt_source_t(opt_pose_list[pp.source_id][0],
                                   opt_pose_list[pp.source_id][1],
                                   opt_pose_list[pp.source_id][2]);
      Eigen::Quaterniond opt_source_q(
          opt_pose_list[pp.source_id][6], opt_pose_list[pp.source_id][3],
          opt_pose_list[pp.source_id][4], opt_pose_list[pp.source_id][5]);
      source_normal = opt_source_q * source_normal;
      source_centriod = opt_source_q * source_centriod + opt_source_t;

      Eigen::Vector3d target_normal(pp.target_plane.normal_x,
                                    pp.target_plane.normal_y,
                                    pp.target_plane.normal_z);
      Eigen::Vector3d target_centriod(pp.target_plane.x, pp.target_plane.y,
                                      pp.target_plane.z);
      Eigen::Vector3d opt_target_t(opt_pose_list[pp.target_id][0],
                                   opt_pose_list[pp.target_id][1],
                                   opt_pose_list[pp.target_id][2]);
      Eigen::Quaterniond opt_target_q(
          opt_pose_list[pp.target_id][6], opt_pose_list[pp.target_id][3],
          opt_pose_list[pp.target_id][4], opt_pose_list[pp.target_id][5]);
      target_normal = opt_target_q * target_normal;
      target_centriod = opt_target_q * target_centriod + opt_target_t;
      if ((pp.source_id - pp.target_id) == 1) {

        double dis = fabs(target_normal.transpose() *
                          (source_centriod - target_centriod));
        double normal_diff = std::min((source_normal + target_normal).norm(),
                                      (source_normal - target_normal).norm());
        near_dis_residual += dis;
        near_normal_residual += normal_diff;
        near_cnt++;
      } else {
        double dis = fabs(target_normal.transpose() *
                          (source_centriod - target_centriod));
        double normal_diff = std::min((source_normal + target_normal).norm(),
                                      (source_normal - target_normal).norm());
        loop_dis_residual += dis;
        loop_normal_residual += normal_diff;
        loop_cnt++;
      }
    }
    std::cout << "residual after opt, near dis: "
              << near_dis_residual / near_cnt
              << ", near normal: " << near_normal_residual / near_cnt
              << std::endl;
    std::cout << "residual after opt, loop dis: "
              << loop_dis_residual / loop_cnt
              << ", near normal: " << loop_normal_residual / loop_cnt
              << std::endl;
  }

  for (size_t i = 0; i < key_frame_last; i++) {
    sensor_msgs::PointCloud2 pub_cloud;
    pcl::toROSMsg(*key_cloud_list[i], pub_cloud);
    pub_cloud.header.frame_id = "camera_init";
    pubFirstCloud.publish(pub_cloud);
    late_loop.sleep();
  }
  std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> correct_cloud_list;
  for (size_t i = key_frame_last; i < opt_pose_list.size(); i++) {
    pcl::PointCloud<pcl::PointXYZI> correct_cloud;
    Eigen::Vector3d opt_translation(opt_pose_list[i][0], opt_pose_list[i][1],
                                    opt_pose_list[i][2]);
    Eigen::Quaterniond opt_q(opt_pose_list[i][6], opt_pose_list[i][3],
                             opt_pose_list[i][4], opt_pose_list[i][5]);
    for (size_t j = 0; j < key_cloud_list2[i - key_frame_last]->size(); j++) {
      pcl::PointXYZI pi = key_cloud_list2[i - key_frame_last]->points[j];
      Eigen::Vector3d pv(pi.x, pi.y, pi.z);
      // back projection
      pv = origin_pose_list[i].second.transpose() * pv -
           origin_pose_list[i].second.transpose() * origin_pose_list[i].first;
      // re-projection
      pv = opt_q * pv + opt_translation;
      pi.x = pv[0];
      pi.y = pv[1];
      pi.z = pv[2];
      correct_cloud.push_back(pi);
    }
    pcl::PointXYZI pose_point;
    pose_point.x = opt_translation[0];
    pose_point.y = opt_translation[1];
    pose_point.z = opt_translation[2];
    pose_cloud->push_back(pose_point);

    sensor_msgs::PointCloud2 pub_cloud;
    pcl::toROSMsg(correct_cloud, pub_cloud);
    pub_cloud.header.frame_id = "camera_init";
    pubCorrectCloud.publish(pub_cloud);
    late_loop.sleep();
    down_sampling_voxel(correct_cloud, 0.25);
    correct_cloud_list.push_back(correct_cloud.makeShared());

    nav_msgs::Odometry odom;
    odom.header.frame_id = "camera_init";
    odom.pose.pose.position.x = opt_translation[0];
    odom.pose.pose.position.y = opt_translation[1];
    odom.pose.pose.position.z = opt_translation[2];
    odom.pose.pose.orientation.w = opt_q.w();
    odom.pose.pose.orientation.x = opt_q.x();
    odom.pose.pose.orientation.y = opt_q.y();
    odom.pose.pose.orientation.z = opt_q.z();
    pubOdomCorreted.publish(odom);
    loop.sleep();
  }

  // calc gt
  if (calc_gt_enable) {
    std::ofstream gt_file(loop_gt_file);
    std::cout << "calc gt for loop!" << std::endl;
    pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kd_tree(
        new pcl::KdTreeFLANN<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI> history_pose_cloud;
    for (size_t i = 0; i < key_cloud_list.size(); i++) {
      pcl::PointXYZI pi;
      pi.x = opt_pose_list[i][0];
      pi.y = opt_pose_list[i][1];
      pi.z = opt_pose_list[i][2];
      history_pose_cloud.push_back(pi);
      gt_file << i << "," << pi.x << "," << pi.y << "," << pi.z << "," << 0
              << "," << 0 << "," << 0 << std::endl;
    }
    kd_tree->setInputCloud(history_pose_cloud.makeShared());
    std::vector<int> indices;
    std::vector<float> distances;
    double radius = 20;
    double overlap_threshold = 0.3;
    int gt_loop_num = 0;
    for (size_t i = key_frame_last; i < opt_pose_list.size(); i++) {
      double max_overlap = 0;
      bool trigger_loop = false;
      int loop_id = 0;
      pcl::PointXYZI searchPoint;
      searchPoint.x = opt_pose_list[i][0];
      searchPoint.y = opt_pose_list[i][1];
      searchPoint.z = opt_pose_list[i][2];
      int size = kd_tree->radiusSearch(searchPoint, radius, indices, distances);
      for (int j = 0; j < size; j++) {
        pcl::PointCloud<pcl::PointXYZI> ds_cloud1 =
            *correct_cloud_list[i - key_frame_last];
        pcl::PointCloud<pcl::PointXYZI> ds_cloud2 = *key_cloud_list[indices[j]];
        double overlap =
            calc_overlap(ds_cloud1.makeShared(), ds_cloud2.makeShared(), 0.5);
        if (overlap > max_overlap) {
          max_overlap = overlap;
          loop_id = indices[j];
        }
      }
      if (max_overlap > overlap_threshold) {
        trigger_loop = true;
        gt_loop_num++;
        sensor_msgs::PointCloud2 pub_cloud;
        pcl::toROSMsg(*correct_cloud_list[i - key_frame_last], pub_cloud);
        pub_cloud.header.frame_id = "camera_init";
        pubCureentCloud.publish(pub_cloud);
        loop.sleep();
        pcl::toROSMsg(*key_cloud_list[loop_id], pub_cloud);
        pub_cloud.header.frame_id = "camera_init";
        pubMatchedCloud.publish(pub_cloud);
        loop.sleep();
        gt_file << i << "," << searchPoint.x << "," << searchPoint.y << ","
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
        gt_file << i << "," << searchPoint.x << "," << searchPoint.y << ","
                << searchPoint.z << "," << 0 << "," << 0 << "," << 0
                << std::endl;
      }
    }
  }
  return 0;
}