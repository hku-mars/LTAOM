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
  std::string result_path =
      "/home/ycj/matlab_code/loop_detection/result/" + data_name + "/" +
      data_name + "_" + icp_string.substr(0, icp_string.find(".") + 3) + ".txt";
  std::ofstream result_file(result_path);
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
  pcl::PointCloud<pcl::PointXYZI>::Ptr current_key_cloud(
      new pcl::PointCloud<pcl::PointXYZI>);
  bool is_build_descriptor = false;
  int key_frame_id = 0;
  int count = 0;

  for (size_t i = 0; i < frame_number_list.size(); i++) {
    if (count == 0) {
      pcl::PointCloud<pcl::PointXYZI>::Ptr temp_cloud(
          new pcl::PointCloud<pcl::PointXYZI>);
      key_cloud_list.push_back(temp_cloud);
      current_key_cloud->clear();
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
    down_sampling_voxel(*register_cloud, config_setting.ds_size_);
    for (size_t i = 0; i < register_cloud->size(); i++) {
      // key_cloud_list.back()->points.push_back(register_cloud->points[i]);
      current_key_cloud->points.push_back(register_cloud->points[i]);
    }
    if (count == config_setting.sub_frame_num_ / 2) {
      current_mean_position = translation;
    }
    if (count < config_setting.sub_frame_num_ - 1) {
      count++;
    } else {
      count = 0;
      is_build_descriptor = true;
    }
    if (is_build_descriptor) {
      std::cout << std::endl;
      std::cout << "Key Frame:" << key_frame_id
                << ", cloud size:" << current_key_cloud->size() << std::endl;
      debug_file << std::endl;
      debug_file << "Key frame:" << key_frame_id
                 << ", cloud size:" << current_key_cloud->size() << std::endl;

      result_file << key_frame_id << "," << current_mean_position[0] << ","
                  << current_mean_position[1] << "," << current_mean_position[2]
                  << ",";
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
      // binary_extractor_debug(config_setting, merge_plane_list,
      //                        key_cloud_list[key_frame_id], binary_list,
      //                        binary_around_list);
      // std::cout << "binary around size:" << binary_around_list.size()
      //           << std::endl;

      binary_extractor(config_setting, merge_plane_list, current_key_cloud,
                       binary_list);

      history_binary_list.push_back(binary_list);

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
      int best_frame = -1;
      for (int i = 0; i < alternative_match.size(); i++) {
        if (alternative_match[i].match_list_.size() > 0) {
          bool fine_sucess = false;
          Eigen::Matrix3d std_rot;
          Eigen::Vector3d std_t;
#ifdef debug
          debug_file << "[Loop Sucess] " << key_frame_id << "--" << match_frame
                     << ", candidate id:" << candidate_id
                     << ", icp:" << best_score << std::endl;
          debug_file << "[Loop Info] "
                     << "rough size:" << rough_size
                     << ", match size:" << match_size
                     << ", rough triangle dis:" << outlier_mean_triangle_dis
                     << ", fine triangle dis:" << mean_triangle_dis
                     << ", rough binary similarity:"
                     << outlier_mean_binary_similarity
                     << ", fine binary similarity:" << mean_binary_similarity
                     << std::endl;
#endif
          sucess_match_list.clear();
          fine_loop_detection_tbb(
              config_setting, alternative_match[i].match_list_, fine_sucess,
              std_rot, std_t, sucess_match_list, unsucess_match_list);
          if (fine_sucess) {
            double score = geometric_verify(
                frame_plane_cloud,
                history_plane_list[alternative_match[i].match_frame_], std_rot,
                std_t);
            // debug_file << "Fine sucess. Icp score:" << score << std::endl;
            if (score > best_score) {
              unsucess_match_list_publish = unsucess_match_list;
              sucess_match_list_publish = sucess_match_list;
              best_frame = alternative_match[i].match_frame_;
              best_score = score;
              best_rot = std_rot;
              best_t = std_t;
              rough_size = alternative_match[i].match_list_.size();
              match_size = sucess_match_list.size();
              candidate_id = i;
            }
          }
        }
      }
      if (best_score > icp_threshold) {
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
      auto t4 = std::chrono::high_resolution_clock::now();
      is_build_descriptor = false;
      auto t_fine_loop_end = std::chrono::high_resolution_clock::now();

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
      auto t_add_descriptor_begin = std::chrono::high_resolution_clock::now();
      add_STD(STD_map, STD_list);
      auto t_add_descriptor_end = std::chrono::high_resolution_clock::now();
      mean_time += time_inc(t_build_descriptor_end, t_build_descriptor_begin) +
                   time_inc(t_candidate_search_end, t_candidate_search_begin) +
                   time_inc(t_fine_loop_end, t_fine_loop_begin) +
                   time_inc(t_add_descriptor_end, t_add_descriptor_begin);
      debug_file << "[Time] build_descriptor:"
                 << time_inc(t_build_descriptor_end, t_build_descriptor_begin)
                 << ", candidate search:"
                 << time_inc(t_candidate_search_end, t_candidate_search_begin)
                 << ", fine loop detect:"
                 << time_inc(t_fine_loop_end, t_fine_loop_begin)
                 << ", add descriptor:"
                 << time_inc(t_add_descriptor_end, t_add_descriptor_begin)
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
        std::cout << "Binary size:" << history_binary_list[key_frame_id].size()
                  << ", augument size:" << binary_augument_num
                  << ", augument rate:"
                  << binary_augument_num * 1.0 /
                         history_binary_list[key_frame_id].size()
                  << std::endl;
        result_file
            << history_binary_list[key_frame_id].size() << ","
            << binary_augument_num << "," << best_score << ","
            << time_inc(t_build_descriptor_end, t_build_descriptor_begin) << ","
            << time_inc(t_candidate_search_end, t_candidate_search_begin) << ","
            << time_inc(t_fine_loop_end, t_fine_loop_begin) << ","
            << time_inc(t_add_descriptor_end, t_add_descriptor_begin)
            << std::endl;

        int std_augument_num = 0;
        double mean_dis = 0;
        double mean_similarity = 0;

        mean_dis = mean_dis / std_augument_num;
        mean_similarity = mean_similarity / std_augument_num;
        std::cout << "STD size:" << STD_list.size()
                  << ", augument size:" << std_augument_num
                  << ", augument rate:"
                  << std_augument_num * 1.0 / STD_list.size()
                  << ", mean dis:" << mean_dis
                  << ", mean similarity:" << mean_similarity << std::endl;
        // publish_std_list(history_STD_list[key_frame_id], pubSTD);
        // getchar();
      } else {
        result_file
            << 0 << "," << 0 << "," << 0 << "," << 0 << "," << 0 << "," << 0
            << "," << 0 << "," << 0 << "," << 0 << "," << 0 << "," << 0 << ","
            << time_inc(t_build_descriptor_end, t_build_descriptor_begin) << ","
            << time_inc(t_candidate_search_end, t_candidate_search_begin) << ","
            << time_inc(t_fine_loop_end, t_fine_loop_begin) << ","
            << time_inc(t_add_descriptor_end, t_add_descriptor_begin)
            << std::endl;
        // getchar();
      }
      down_sampling_voxel(*current_key_cloud, 1);
      for (size_t i = 0; i < current_key_cloud->size(); i++) {
        key_cloud_list.back()->push_back(current_key_cloud->points[i]);
      }
      key_frame_id++;
      // if (key_frame_id >= 230) {
      //   break;
      // }
      for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++) {
        delete (iter->second);
      }
    }
  }
  // save pcd
  // std::cout << "saving pcd......" << std::endl;
  // for (size_t i = 0; i < key_cloud_list.size(); i++) {
  //   std::string pcd_file =
  //       "/data/pcds/kitti360-02/" + std::to_string(i) + ".pcd";
  //   key_cloud_list[i]->height = 1;
  //   key_cloud_list[i]->width = key_cloud_list[i]->points.size();
  //   pcl::io::savePCDFileASCII(pcd_file, *key_cloud_list[i]);
  // }
  // std::cout << "saving pcd finish!" << std::endl;

  ros::shutdown();
  return 0;
}