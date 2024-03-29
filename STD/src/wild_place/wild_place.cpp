#include "include/std.h"
#include <nav_msgs/Odometry.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud2.h>

// #define debug

void load_wild_pose_with_time(
    const std::string &pose_file,
    std::vector<std::pair<Eigen::Vector3d, Eigen::Matrix3d>> &pose_list,
    std::vector<std::string> &time_list) {
  time_list.clear();
  pose_list.clear();
  std::ifstream fin(pose_file);
  std::string line;
  Eigen::Matrix<double, 1, 7> temp_matrix;
  // 跳过第一行
  // getline(fin, line);
  while (getline(fin, line)) {
    std::istringstream sin(line);
    std::vector<std::string> Waypoints;
    std::string info;
    int number = 0;
    while (getline(sin, info, ',')) {
      if (number == 0) {
        double time;
        std::stringstream data;
        data << info;
        data >> time;
        time_list.push_back(data.str());
        number++;
      } else {
        double p;
        std::stringstream data;
        data << info;
        data >> p;
        temp_matrix[number - 1] = p;
        if (number == 7) {
          Eigen::Vector3d translation(temp_matrix[0], temp_matrix[1],
                                      temp_matrix[2]);
          Eigen::Quaterniond q(temp_matrix[6], temp_matrix[3], temp_matrix[4],
                               temp_matrix[5]);
          std::pair<Eigen::Vector3d, Eigen::Matrix3d> single_pose;
          single_pose.first = translation;
          single_pose.second = q.toRotationMatrix();
          pose_list.push_back(single_pose);
        }
        number++;
      }
    }
  }
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "std_loop");
  ros::NodeHandle nh;
  std::string data_name = "";
  std::string setting_path = "";
  std::string pcds_file = "";
  std::string pose_file = "";
  std::string loop_gt_file = "";
  double icp_threshold = 0.5;
  bool calc_gt_enable = false;
  nh.param<std::string>("data_name", data_name, "");
  nh.param<std::string>("setting_path", setting_path, "");
  nh.param<std::string>("pcds_file", pcds_file, "");
  nh.param<std::string>("pose_file", pose_file, "");
  nh.param<std::string>("loop_gt_file", loop_gt_file, "");
  nh.param<bool>("calc_gt_enable", calc_gt_enable, false);
  nh.param<double>("icp_threshold", icp_threshold, 0.5);
  std::string icp_string = std::to_string(icp_threshold);
  std::string result_path =
      "/home/ycj/matlab_code/loop_detection/result/std_pr/" + data_name +
      ".txt";
  // std::string result_path =
  //     "/home/ycj/matlab_code/loop_detection/result/" + data_name + "/" +
  //     data_name + "_" + icp_string.substr(0, icp_string.find(".") + 3) +
  //     ".txt";
  std::ofstream result_file(result_path);
  std::ofstream debug_file("/home/ycj/catkin_ws/src/STD/Log/log.txt");

  // publsiher
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
  ros::Rate slow_loop(50000);
  ConfigSetting config_setting;
  load_config_setting(setting_path, config_setting);
  std::vector<std::pair<Eigen::Vector3d, Eigen::Matrix3d>> pose_list;
  std::vector<std::string> time_list;
  load_wild_pose_with_time(pose_file, pose_list, time_list);
  std::cout << "load wild poses:" << pose_list.size() << std::endl;
  // getchar();

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

  bool is_build_descriptor = false;
  int key_frame_id = 0;
  int count = 0;
  double mean_time = 0;
  pcl::PointCloud<pcl::PointXYZI>::Ptr pose_cloud(
      new pcl::PointCloud<pcl::PointXYZI>);

  int start_frame = 0;
  // for (size_t key_frame_id = start_frame; key_frame_id < time_list.size();
  //      key_frame_id++) {
  //   std::string single_pcd_file = pcds_file + time_list[key_frame_id] +
  //   ".pcd";
  //   // load lidar point cloud and start loop
  //   pcl::PointCloud<pcl::PointXYZI>::Ptr current_key_cloud(
  //       new pcl::PointCloud<pcl::PointXYZI>);
  //   pcl::PointCloud<pcl::PointXYZI> lidar_cloud;
  //   sensor_msgs::PointCloud2 cloud_msg;
  //   auto t_load_begin = std::chrono::high_resolution_clock::now();
  //   pcl::io::loadPCDFile(single_pcd_file, cloud_msg);
  //   pcl::fromROSMsg(cloud_msg, *current_key_cloud);
  //   auto t_load_end = std::chrono::high_resolution_clock::now();
  //   down_sampling_voxel(*current_key_cloud, 0.5);
  //   std::cout << std::endl;
  //   std::cout << "Key Frame:" << key_frame_id
  //             << ", cloud size:" << current_key_cloud->size() << std::endl;
  //   debug_file << std::endl;
  //   debug_file << "Key frame:" << key_frame_id
  //              << ", cloud size:" << current_key_cloud->size() << std::endl;
  //   std::cout << "load cloud time:" << time_inc(t_load_end, t_load_begin)
  //             << " ms" << std::endl;
  //   for (size_t i = 0; i < current_key_cloud->size(); i++) {
  //     Eigen::Vector3d pv = point2vec(current_key_cloud->points[i]);
  //     pv = pose_list[key_frame_id].second * pv +
  //     pose_list[key_frame_id].first; current_key_cloud->points[i] =
  //     vec2point(pv);
  //   }
  //   pcl::PointCloud<pcl::PointXYZI>::Ptr temp_cloud(
  //       new pcl::PointCloud<pcl::PointXYZI>);
  //   for (auto pi : current_key_cloud->points) {
  //     temp_cloud->points.push_back(pi);
  //   }
  //   key_cloud_list.push_back(temp_cloud);
  // }

  for (size_t key_frame_id = start_frame; key_frame_id < time_list.size();
       key_frame_id++) {
    std::string single_pcd_file = pcds_file + time_list[key_frame_id] + ".pcd";
    // load lidar point cloud and start loop
    pcl::PointCloud<pcl::PointXYZI>::Ptr origin_cloud(
        new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr current_key_cloud(
        new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI> lidar_cloud;
    sensor_msgs::PointCloud2 cloud_msg;
    auto t_load_begin = std::chrono::high_resolution_clock::now();
    // pcl::PCDReader reader;
    // pcl::PointCloud<pcl::PointXYZRGB> read_cloud;
    // reader.read(single_pcd_file, read_cloud);
    pcl::io::loadPCDFile(single_pcd_file, cloud_msg);
    pcl::fromROSMsg(cloud_msg, *origin_cloud);
    auto t_load_end = std::chrono::high_resolution_clock::now();
    // for (size_t i = 0; i < read_cloud.size(); i++) {
    //   pcl::PointXYZI pi;
    //   pi.x = read_cloud.points[i].x;
    //   pi.y = read_cloud.points[i].y;
    //   pi.z = read_cloud.points[i].z;
    //   current_key_cloud->push_back(pi);
    // }

    std::cout << std::endl;
    std::cout << "Key Frame:" << key_frame_id
              << ", cloud size:" << origin_cloud->size() << std::endl;
    debug_file << std::endl;
    debug_file << "Key frame:" << key_frame_id
               << ", cloud size:" << origin_cloud->size() << std::endl;
    std::cout << "load cloud time:" << time_inc(t_load_end, t_load_begin)
              << " ms" << std::endl;
    // down_sampling_voxel(lidar_cloud, 0.5);

    sensor_msgs::PointCloud2 pub_cloud;
    // pcl::toROSMsg(*current_key_cloud, pub_cloud);
    // pub_cloud.header.frame_id = "camera_init";
    // pubRegisterCloud.publish(pub_cloud);
    // slow_loop.sleep();
    Eigen::Quaterniond q(pose_list[key_frame_id].second);
    nav_msgs::Odometry odom;
    odom.header.frame_id = "camera_init";
    odom.pose.pose.position.x = pose_list[key_frame_id].first[0];
    odom.pose.pose.position.y = pose_list[key_frame_id].first[1];
    odom.pose.pose.position.z = pose_list[key_frame_id].first[2];
    odom.pose.pose.orientation.w = q.w();
    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();
    pubOdomAftMapped.publish(odom);
    slow_loop.sleep();
    pcl::PointXYZI single_pose_point;
    single_pose_point.x = pose_list[key_frame_id].first[0];
    single_pose_point.y = pose_list[key_frame_id].first[1];
    single_pose_point.z = pose_list[key_frame_id].first[2];
    pose_cloud->points.push_back(single_pose_point);
    // build descriptor
    result_file << key_frame_id << "," << pose_list[key_frame_id].first[0]
                << "," << pose_list[key_frame_id].first[1] << ","
                << pose_list[key_frame_id].first[2] << ",";
    auto t_build_descriptor_begin = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < origin_cloud->size();
         i += config_setting.point_skip_) {
      Eigen::Vector3d pv = point2vec(origin_cloud->points[i]);
      pv = pose_list[key_frame_id].second * pv + pose_list[key_frame_id].first;
      current_key_cloud->push_back(vec2point(pv));
    }
    std::unordered_map<VOXEL_LOC, OctoTree *> voxel_map;
    init_voxel_map(config_setting, *current_key_cloud, voxel_map);
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr frame_plane_cloud(
        new pcl::PointCloud<pcl::PointXYZINormal>);
    get_plane(voxel_map, frame_plane_cloud);
    history_plane_list.push_back(frame_plane_cloud);
    std::vector<Plane *> proj_plane_list;
    std::vector<Plane *> merge_plane_list;
    get_project_plane(config_setting, voxel_map, proj_plane_list);

    if (proj_plane_list.size() == 0) {
      Plane *single_plane = new Plane;
      single_plane->normal_ << 0, 0, 1;
      single_plane->center_ = pose_list[key_frame_id].first;
      merge_plane_list.push_back(single_plane);
    } else {
      sort(proj_plane_list.begin(), proj_plane_list.end(), plane_greater_sort);
      merge_plane(config_setting, proj_plane_list, merge_plane_list);
      sort(merge_plane_list.begin(), merge_plane_list.end(),
           plane_greater_sort);
    }

    std::vector<BinaryDescriptor> binary_list;
    binary_extractor(config_setting, merge_plane_list, current_key_cloud,
                     binary_list);
    std::vector<STD> STD_list;
    generate_std(config_setting, binary_list, key_frame_id, STD_list);
    auto t_build_descriptor_end = std::chrono::high_resolution_clock::now();

    history_binary_list.push_back(binary_list);
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

    debug_file << "[Corner] corner size:" << binary_list.size()
               << "  descriptor size:" << STD_list.size() << std::endl;
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
        // #ifdef debug
        //         debug_file << "[Rough match] rough match frame:"
        //                    << alternative_match[i].match_frame_
        //                    << " match size:" <<
        //                    alternative_match[i].match_list_.size()
        //                    << std::endl;
        // #endif
        sucess_match_list.clear();
        fine_loop_detection_tbb(
            config_setting, alternative_match[i].match_list_, fine_sucess,
            std_rot, std_t, sucess_match_list, unsucess_match_list);
        // fine_loop_detection(config_setting,
        // alternative_match[i].match_list_,
        //                     fine_sucess, std_rot, std_t,
        //                     sucess_match_list, unsucess_match_list);
        if (fine_sucess) {
          double icp_score = geometric_verify(
              config_setting, frame_plane_cloud,
              history_plane_list[alternative_match[i].match_frame_ -
                                 start_frame],
              std_rot, std_t);
          double score = icp_score;
          // double score = icp_score + sucess_match_list.size() * 1.0 / 1000;
          // #ifdef debug
          //           debug_file << "Fine sucess, Fine size:" <<
          //           sucess_match_list.size()
          //                      << "  ,Icp score:" << icp_score << ", score:"
          //                      << score
          //                      << std::endl;
          // #endif
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
            // if (best_score > 0.5) {
            //   break;
            // }
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
      mean_binary_similarity = calc_binary_similaity(sucess_match_list_publish);
      outlier_mean_triangle_dis =
          calc_triangle_dis(unsucess_match_list_publish);
      outlier_mean_binary_similarity =
          calc_binary_similaity(unsucess_match_list_publish);
    } else {
      triggle_loop = false;
    }
    is_build_descriptor = false;
    auto t_fine_loop_end = std::chrono::high_resolution_clock::now();
    if (triggle_loop) {

      debug_file << "[Loop Sucess] " << key_frame_id << "--" << match_frame
                 << ", candidate id:" << candidate_id << ", icp:" << best_score
                 << std::endl;
      debug_file << "[Loop Info] "
                 << "rough size:" << rough_size << ", match size:" << match_size
                 << ", rough triangle dis:" << outlier_mean_triangle_dis
                 << ", fine triangle dis:" << mean_triangle_dis
                 << ", rough binary similarity:"
                 << outlier_mean_binary_similarity
                 << ", fine binary similarity:" << mean_binary_similarity
                 << std::endl;
#ifdef debug
#endif
      result_file << 1 << "," << match_frame << "," << candidate_id << ","
                  << match_size << "," << rough_size << ","
                  << loop_translation[0] << "," << loop_translation[1] << ","
                  << loop_translation[2] << ",";

      pcl::toROSMsg(*key_cloud_list[match_frame - start_frame], pub_cloud);
      pub_cloud.header.frame_id = "camera_init";
      pubMatchedCloud.publish(pub_cloud);
      loop.sleep();
      pcl::PointCloud<pcl::PointXYZ> matched_key_points_cloud;
      for (auto var : history_binary_list[match_frame - start_frame]) {
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
      publish_binary(history_binary_list[match_frame - start_frame], color2,
                     "history", pubSTD);
      loop.sleep();
      publish_std(sucess_match_list_publish, pubSTD);
    } else {
      debug_file << "[Loop Fail] " << key_frame_id << ", icp:" << best_score
                 << std::endl;
    }
    auto t_add_descriptor_begin = std::chrono::high_resolution_clock::now();
    if (key_frame_id % config_setting.std_add_skip_frame_ == 0)
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
      down_sampling_voxel(*current_key_cloud, 0.5);
      double cloud_overlap =
          calc_overlap(current_key_cloud, key_cloud_list[match_frame], 0.5);
      result_file << history_binary_list[key_frame_id].size() << ","
                  << cloud_overlap << "," << best_score << ","
                  << time_inc(t_build_descriptor_end, t_build_descriptor_begin)
                  << ","
                  << time_inc(t_candidate_search_end, t_candidate_search_begin)
                  << "," << time_inc(t_fine_loop_end, t_fine_loop_begin) << ","
                  << time_inc(t_add_descriptor_end, t_add_descriptor_begin)
                  << std::endl;

      int std_augument_num = 0;
      double mean_dis = 0;
      double mean_similarity = 0;
      mean_dis = mean_dis / std_augument_num;
      mean_similarity = mean_similarity / std_augument_num;
    } else {
      down_sampling_voxel(*current_key_cloud, 0.5);
      result_file << 0 << "," << 0 << "," << 0 << "," << 0 << "," << 0 << ","
                  << 0 << "," << 0 << "," << 0 << "," << 0 << "," << 0 << ","
                  << 0 << ","
                  << time_inc(t_build_descriptor_end, t_build_descriptor_begin)
                  << ","
                  << time_inc(t_candidate_search_end, t_candidate_search_begin)
                  << "," << time_inc(t_fine_loop_end, t_fine_loop_begin) << ","
                  << time_inc(t_add_descriptor_end, t_add_descriptor_begin)
                  << std::endl;
      // getchar();
    }

    // down_sampling_voxel(*current_key_cloud, 0.5);
    pcl::PointCloud<pcl::PointXYZI>::Ptr temp_cloud(
        new pcl::PointCloud<pcl::PointXYZI>);
    for (auto pi : current_key_cloud->points) {
      temp_cloud->points.push_back(pi);
    }
    key_cloud_list.push_back(temp_cloud);
    for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++) {
      delete (iter->second);
    }
  }

  if (calc_gt_enable) {
    std::ofstream gt_file(loop_gt_file);
    std::cout << "calc gt for loop!" << std::endl;
    pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kd_tree(
        new pcl::KdTreeFLANN<pcl::PointXYZI>);
    kd_tree->setInputCloud(pose_cloud);
    std::vector<int> indices;
    std::vector<float> distances;
    double radius = 15;
    double overlap_threshold = 0.5;
    int gt_loop_num = 0;
    for (int i = 0; i < pose_cloud->size(); i++) {
      std::cout << "i:" << i << std::endl;
      double max_overlap = 0;
      bool trigger_loop = false;
      int loop_id = 0;
      pcl::PointXYZI searchPoint = pose_cloud->points[i];
      int size = kd_tree->radiusSearch(searchPoint, radius, indices, distances);
      for (int j = 0; j < size; j++) {
        if (indices[j] >= i - 100) {
          continue;
        } else {
          pcl::PointCloud<pcl::PointXYZI> ds_cloud1 = *key_cloud_list[i];
          pcl::PointCloud<pcl::PointXYZI> ds_cloud2 =
              *key_cloud_list[indices[j]];
          double overlap = calc_overlap(ds_cloud1.makeShared(),
                                        ds_cloud2.makeShared(), 0.5, 2);
          if (overlap > max_overlap) {
            max_overlap = overlap;
            loop_id = indices[j];
            if (max_overlap > overlap_threshold) {
              break;
            }
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
        // std::string max_overlap_str = std::to_string(max_overlap);
        // max_overlap_str =
        //     max_overlap_str.substr(0, max_overlap_str.find(".") + 4);
        // max_overlap_str = "Overlap: " + max_overlap_str;
        // // publish_map(pubLaserCloudMap);
        // cv::Mat max_overlap_pic = cv::Mat::zeros(200, 800, CV_8UC3);
        // cv::Point siteNO;
        // siteNO.x = 100;
        // siteNO.y = 100;
        // cv::putText(max_overlap_pic, max_overlap_str, siteNO, 4, 2,
        //             cv::Scalar(255, 255, 255), 4);
        // cv::imshow("", max_overlap_pic);
        // cv::waitKey(500);
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