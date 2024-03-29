#include "include/std.h"
#include <nav_msgs/Odometry.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud2.h>

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
  std::string bag_file = "";
  std::string pose_file = "";
  std::string loop_gt_file = "";
  double icp_threshold = 0.5;
  bool calc_gt_enable = false;
  nh.param<std::string>("data_name", data_name, "");
  nh.param<std::string>("setting_path", setting_path, "");
  nh.param<std::string>("bag_file", bag_file, "");
  nh.param<std::string>("pose_file", pose_file, "");
  nh.param<std::string>("loop_gt_file", loop_gt_file, "");
  nh.param<bool>("calc_gt_enable", calc_gt_enable, false);
  nh.param<double>("icp_threshold", icp_threshold, 0.5);
  std::string icp_string = std::to_string(icp_threshold);
  std::string result_path =
      "/home/ycj/matlab_code/loop_detection/result/" + data_name + "/" +
      data_name + "_" + icp_string.substr(0, icp_string.find(".") + 3) + ".txt";
  std::ofstream result_file(result_path);
  std::ofstream gt_file(loop_gt_file);
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
  if (config_setting.gt_file_style_ == 0) {
    load_pose_with_time(pose_file, pose_list, time_list);
  } else if (config_setting.gt_file_style_ == 1) {
    load_cu_pose_with_time(pose_file, pose_list, time_list);
  }

  std::string print_msg = "Sucessfully load pose file:" + pose_file +
                          ". pose size:" + std::to_string(time_list.size());
  ROS_INFO_STREAM(print_msg.c_str());

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

  long current_time = 0;

  // load lidar point cloud and start loop
  pcl::PointCloud<pcl::PointXYZI>::Ptr pose_cloud(
      new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr current_key_cloud(
      new pcl::PointCloud<pcl::PointXYZI>);
  bool is_build_descriptor = false;
  int key_frame_id = 0;
  std::fstream file_;
  file_.open(bag_file, std::ios::in);
  if (!file_) {
    std::cout << "File " << bag_file << " does not exit" << std::endl;
  }
  ROS_INFO("Start to load the rosbag %s", bag_file.c_str());
  rosbag::Bag bag;
  try {
    bag.open(bag_file, rosbag::bagmode::Read);
  } catch (rosbag::BagException e) {
    ROS_ERROR_STREAM("LOADING BAG FAILED: " << e.what());
  }
  std::vector<std::string> types;
  types.push_back(std::string("sensor_msgs/PointCloud2"));
  rosbag::View view(bag, rosbag::TypeQuery(types));
  bool is_init_bag = false;
  bool is_skip_frame = false;
  Eigen::Vector3d init_translation;
  Eigen::Vector3d last_translation;
  Eigen::Quaterniond last_q;
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
      Eigen::Vector3d translation = pose_list[pose_index].first;
      Eigen::Matrix3d rotation = pose_list[pose_index].second;
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
      std::cout << std::endl;
      std::cout << "Key Frame:" << key_frame_id
                << ", cloud size:" << current_key_cloud->size() << std::endl;
      debug_file << std::endl;
      debug_file << "Key frame:" << key_frame_id
                 << ", cloud size:" << current_key_cloud->size() << std::endl;
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

      auto t_descriptor_begin = std::chrono::high_resolution_clock::now();
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
      // binary_extractor_debug(config_setting, merge_plane_list,
      //                        key_cloud_list[key_frame_id], binary_list,
      //                        binary_around_list);
      // std::cout << "binary around size:" << binary_around_list.size()
      //           << std::endl;

      binary_extractor(config_setting, merge_plane_list, current_key_cloud,
                       binary_list);
      std::vector<STD> STD_list;
      generate_std(config_setting, binary_list, key_frame_id, STD_list);
      auto t_descriptor_end = std::chrono::high_resolution_clock::now();

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
      pcl::PointCloud<pcl::PointXYZ> around_key_points_cloud;
      for (auto var : binary_around_list) {
        pcl::PointXYZ pi;
        pi.x = var.location_[0];
        pi.y = var.location_[1];
        pi.z = var.location_[2];
        around_key_points_cloud.push_back(pi);
      }
      pcl::toROSMsg(around_key_points_cloud, pub_cloud);
      pub_cloud.header.frame_id = "camera_init";
      pubMatchedBinary.publish(pub_cloud);
      Eigen::Vector3d color2(0, 1, 0);
      publish_binary(binary_around_list, color2, "history", pubSTD);

      // history_STD_list.push_back(STD_list);

      // debug for binary and std augument
      double dis_threshold = 1.0;
      int binary_augument_num = 0;
      if (key_frame_id >= 1) {
        for (size_t i = 0; i < history_binary_list[key_frame_id].size(); i++) {
          BinaryDescriptor binary1 = history_binary_list[key_frame_id][i];
          for (size_t j = 0; j < history_binary_list[key_frame_id - 1].size();
               j++) {
            BinaryDescriptor binary2 = history_binary_list[key_frame_id - 1][j];
            if ((binary1.location_ - binary2.location_).norm() <
                dis_threshold) {
              binary_augument_num++;
              break;
            }
          }
        }
      }
      debug_augment << key_frame_id << ","
                    << history_binary_list[key_frame_id].size() << ","
                    << binary_augument_num << std::endl;
      // std::cout << "Binary size:" << history_binary_list[key_frame_id].size()
      //           << ", augument size:" << binary_augument_num
      //           << ", augument rate:"
      //           << binary_augument_num * 1.0 /
      //                  history_binary_list[key_frame_id].size()
      //           << std::endl;
      int std_augument_num = 0;
      // if (key_frame_id >= 1) {
      //   for (size_t i = 0; i < history_STD_list[key_frame_id].size(); i++) {
      //     STD std1 = history_STD_list[key_frame_id][i];
      //     for (size_t j = 0; j < history_STD_list[key_frame_id - 1].size();
      //          j++) {
      //       STD std2 = history_STD_list[key_frame_id - 1][j];
      //       double dis1 =
      //           (std1.binary_A_.location_ - std2.binary_A_.location_).norm();
      //       double dis2 =
      //           (std1.binary_B_.location_ - std2.binary_B_.location_).norm();
      //       double dis3 =
      //           (std1.binary_C_.location_ - std2.binary_C_.location_).norm();
      //       if (dis1 < dis_threshold && dis2 < dis_threshold &&
      //           dis3 < dis_threshold) {
      //         std_augument_num++;
      //         break;
      //       }
      //     }
      //   }
      // }
      // debug_augment << history_STD_list[key_frame_id].size() << ","
      //               << std_augument_num << std::endl;

      // std::cout << "STD size:" << history_STD_list[key_frame_id].size()
      //           << ", augument size:" << std_augument_num
      //           << ", augument rate : "
      //           << std_augument_num * 1.0 /
      //                  history_STD_list[key_frame_id].size()
      //           << std::endl;
      // // publish_std_list(history_STD_list[key_frame_id], pubSTD);

      debug_file << "[Corner] corner size:" << binary_list.size()
                 << "  descriptor size:" << STD_list.size() << std::endl;
      // candidate search
      auto t_candidate_search_begin = std::chrono::high_resolution_clock::now();
      std::vector<STDMatchList> alternative_match;
      candidate_searcher(config_setting, STD_map, STD_list, alternative_match);
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
          // debug_file << "[Rough match] rough match frame:"
          //            << alternative_match[i].match_frame_ << " match size:"
          //            << alternative_match[i].match_list_.size() << std::endl;
          sucess_match_list.clear();
          fine_loop_detection_tbb(
              config_setting, alternative_match[i].match_list_, fine_sucess,
              std_rot, std_t, sucess_match_list, unsucess_match_list);
          // fine_loop_detection(config_setting,
          // alternative_match[i].match_list_,
          //                     fine_sucess, std_rot, std_t, sucess_match_list,
          //                     unsucess_match_list);
          if (fine_sucess) {
            double icp_score = geometric_verify(
                config_setting, frame_plane_cloud,
                history_plane_list[alternative_match[i].match_frame_], std_rot,
                std_t);
            double score = icp_score + sucess_match_list.size() * 1.0 / 1000;
            // debug_file << "Fine sucess, Fine size:" <<
            // sucess_match_list.size()
            //            << "  ,Icp score:" << icp_score << ", score:" << score
            //            << std::endl;
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

      // publish cloud and descriptor
      // sensor_msgs::PointCloud2 pub_cloud;
      // pcl::toROSMsg(*key_cloud_list[key_frame_id], pub_cloud);
      // pub_cloud.header.frame_id = "camera_init";
      // pubCureentCloud.publish(pub_cloud);
      // loop.sleep();
      // pcl::PointCloud<pcl::PointXYZ> key_points_cloud;
      // for (auto var : binary_list) {
      //   pcl::PointXYZ pi;
      //   pi.x = var.location_[0];
      //   pi.y = var.location_[1];
      //   pi.z = var.location_[2];
      //   key_points_cloud.push_back(pi);
      // }
      // pcl::toROSMsg(key_points_cloud, pub_cloud);
      // pub_cloud.header.frame_id = "camera_init";
      // pubCurrentBinary.publish(pub_cloud);
      // loop.sleep();
      // Eigen::Vector3d color1(1, 1, 0);
      // publish_binary(binary_list, color1, "current", pubSTD);

      if (triggle_loop) {
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
        result_file << 1 << "," << match_frame << "," << candidate_id << ","
                    << match_size << "," << rough_size << ","
                    << loop_translation[0] << "," << loop_translation[1] << ","
                    << loop_translation[2] << ",";

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
      } else {
        debug_file << "[Loop Fail] " << key_frame_id << ", icp:" << best_score
                   << std::endl;
      }
      auto t_add_std_begin = std::chrono::high_resolution_clock::now();
      add_STD(STD_map, STD_list);
      auto t_add_std_end = std::chrono::high_resolution_clock::now();

      double descriptor_time = time_inc(t_descriptor_end, t_descriptor_begin);
      double candidate_search_time =
          time_inc(t_candidate_search_end, t_candidate_search_begin);
      double fine_loop_time =
          time_inc(t_candidate_search_end, t_candidate_search_begin);
      double add_descriptor_time = time_inc(t_add_std_end, t_add_std_begin);
      mean_time += descriptor_time + candidate_search_time + fine_loop_time +
                   add_descriptor_time;
      debug_file << "[Time] build_descriptor:" << descriptor_time
                 << ", candidate search:" << candidate_search_time
                 << ", fine loop detect:" << fine_loop_time
                 << ", add descriptor:" << add_descriptor_time
                 << ", average:" << mean_time / (key_frame_id + 1) << std::endl;
      if (triggle_loop) {
        // debug for binary and std augument
        double dis_threshold = 1.0;
        int binary_augument_num = 0;
        if (key_frame_id >= 1) {
          for (size_t i = 0; i < history_binary_list[key_frame_id].size();
               i++) {
            BinaryDescriptor binary1 = history_binary_list[key_frame_id][i];
            for (size_t j = 0; j < history_binary_list[match_frame].size();
                 j++) {
              BinaryDescriptor binary2 = history_binary_list[match_frame][j];
              if ((binary1.location_ - binary2.location_).norm() <
                  dis_threshold) {
                binary_augument_num++;
                break;
              }
            }
          }
        }
        // debug_augment << key_frame_id << ","
        //               << history_binary_list[key_frame_id].size() << ","
        //               << binary_augument_num << ",";
        std::cout << "Binary size:" << history_binary_list[key_frame_id].size()
                  << ", augument size:" << binary_augument_num
                  << ", augument rate:"
                  << binary_augument_num * 1.0 /
                         history_binary_list[key_frame_id].size()
                  << std::endl;
        result_file << history_binary_list[key_frame_id].size() << ","
                    << binary_augument_num << "," << best_score << ","
                    << descriptor_time << "," << candidate_search_time << ","
                    << fine_loop_time << "," << add_descriptor_time
                    << std::endl;

        int std_augument_num = 0;
        double mean_dis = 0;
        double mean_similarity = 0;
        // if (key_frame_id >= 1) {
        //   for (size_t i = 0; i < history_STD_list[key_frame_id].size(); i++)
        //   {
        //     STD std1 = history_STD_list[key_frame_id][i];
        //     for (size_t j = 0; j < history_STD_list[match_frame].size(); j++)
        //     {
        //       STD std2 = history_STD_list[match_frame][j];
        //       double dis1 =
        //           (std1.binary_A_.location_ -
        //           std2.binary_A_.location_).norm();
        //       double dis2 =
        //           (std1.binary_B_.location_ -
        //           std2.binary_B_.location_).norm();
        //       double dis3 =
        //           (std1.binary_C_.location_ -
        //           std2.binary_C_.location_).norm();
        //       if (dis1 < dis_threshold && dis2 < dis_threshold &&
        //           dis3 < dis_threshold) {
        //         mean_dis += (std1.triangle_ - std2.triangle_).norm() /
        //                     std1.triangle_.norm();
        //         double sim1 = binary_similarity(std1.binary_A_,
        //         std2.binary_A_); double sim2 =
        //         binary_similarity(std1.binary_B_, std2.binary_B_); double
        //         sim3 = binary_similarity(std1.binary_C_, std2.binary_C_);
        //         mean_similarity += ((sim1 + sim2 + sim3) / 3);
        //         std_augument_num++;
        //         break;
        //       }
        //     }
        //   }
        // }
        // debug_augment << history_STD_list[key_frame_id].size() << ","
        //               << std_augument_num << std::endl;
        mean_dis = mean_dis / std_augument_num;
        mean_similarity = mean_similarity / std_augument_num;
        std::cout << "STD size:" << STD_list.size()
                  << ", augument size:" << std_augument_num
                  << ", augument rate:"
                  << std_augument_num * 1.0 / STD_list.size()
                  << ", mean dis:" << mean_dis
                  << ", mean similarity:" << mean_similarity << std::endl;
        // publish_std_list(history_STD_list[key_frame_id], pubSTD);
        // if (key_frame_id > 490)
        // getchar();
      } else {
        result_file << 0 << "," << 0 << "," << 0 << "," << 0 << "," << 0 << ","
                    << 0 << "," << 0 << "," << 0 << "," << 0 << "," << 0 << ","
                    << 0 << "," << descriptor_time << ","
                    << candidate_search_time << "," << fine_loop_time << ","
                    << add_descriptor_time << std::endl;
        // getchar();
      }
      down_sampling_voxel(*current_key_cloud, 0.5);
      for (size_t i = 0; i < current_key_cloud->size(); i++) {
        key_cloud_list.back()->push_back(current_key_cloud->points[i]);
      }
      key_frame_id++;
      for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++) {
        delete (iter->second);
      }
    }
  }

  // for (size_t i = 0; i < key_cloud_list.size(); i++) {
  //   std::string pcd_file = "/data/pcds/park1/" + std::to_string(i) + ".pcd";
  //   key_cloud_list[i]->height = 1;
  //   key_cloud_list[i]->width = key_cloud_list[i]->points.size();
  //   pcl::io::savePCDFileASCII(pcd_file, *key_cloud_list[i]);
  // }

  if (calc_gt_enable) {
    std::cout << "calc gt for loop!" << std::endl;
    pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kd_tree(
        new pcl::KdTreeFLANN<pcl::PointXYZI>);
    kd_tree->setInputCloud(pose_cloud);
    std::vector<int> indices;
    std::vector<float> distances;
    double radius = 30;
    double overlap_threshold = 0.35;
    int gt_loop_num = 0;
    for (int i = 0; i < pose_cloud->size(); i++) {
      double max_overlap = 0;
      bool trigger_loop = false;
      int loop_id = 0;
      pcl::PointXYZI searchPoint = pose_cloud->points[i];
      int size = kd_tree->radiusSearch(searchPoint, radius, indices, distances);
      for (int j = 0; j < size; j++) {
        if (indices[j] >= i - 50) {
          continue;
        } else {
          pcl::PointCloud<pcl::PointXYZI> ds_cloud1 = *key_cloud_list[i];
          pcl::PointCloud<pcl::PointXYZI> ds_cloud2 =
              *key_cloud_list[indices[j]];
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