/*This code is the implementation of our paper "LTA-OM: Long-Term Association
LiDAR-Inertial Odometry and Mapping".

Current Developer: Zuhao Zou < zuhaozou@yahoo.com >

If you use any code of this repo in your academic research, please cite at least
one of our papers:
[1] Zou, Zuhao, et al. "LTA-OM: Long-Term Association LiDAR-Inertial
    Odometry and Mapping"
[2] Yuan, Chongjian, et al. "Std: Stable triangle descriptor for 3d place recognition"
[3] Xu, Wei, et al. "Fast-lio2: Fast direct lidar-inertial odometry."
[4] Xu, Wei, and Fu Zhang. "Fast-lio: A fast, robust lidar-inertial odometry
    package by tightly-coupled iterated kalman filter."
[5] Cai, Yixi, Wei Xu, and Fu Zhang. "ikd-Tree: An Incremental KD Tree for
    Robotic Applications."
[6] Lin, Jiarong, and Fu Zhang. "Loam-livox: A fast, robust, high-precision
    LiDAR odometry and mapping package for LiDARs of small FoV."

For commercial use, please contact me < zuhaozou@yahoo.com > and
Dr. Fu Zhang < fuzhang@hku.hk >.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.
 3. Neither the name of the copyright holder nor the names of its
    contributors may be used to endorse or promote products derived from this
    software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 POSSIBILITY OF SUCH DAMAGE.
*/

#include "include/std.h"

float ds_size_ = 0.5;
int useful_corner_num_ = 30;

/* for key points*/
float plane_merge_normal_thre_;
float plane_merge_dis_thre_;
float plane_detection_thre_ = 0.01;
float voxel_size_ = 1.0;
int voxel_init_num_ = 10;
int proj_plane_num_ = 3;
float proj_image_resolution_ = 0.5;
float proj_image_high_inc_ = 0.5;
float proj_dis_min_ = 0.2;
float proj_dis_max_ = 5;
float summary_min_thre_ = 10;
bool line_filter_enable_ = false;

/* for STD */
float descriptor_near_num_ = 10;
float descriptor_min_len_ = 1;
float descriptor_max_len_ = 10;
float non_max_suppression_radius_ = 3.0;
float std_side_resolution_ = 0.2;

/* for place recognition*/
int candidate_num_ = 50;
int sub_frame_num_ = 10;
float icp_threshold_ = 0.5;

void load_config_setting(std::string &config_file,
                         ConfigSetting &config_setting) {
  cv::FileStorage fSettings(config_file, cv::FileStorage::READ);
  if (!fSettings.isOpened()) {
    std::cerr << "Failed to open settings file at: " << config_file
              << std::endl;
    exit(-1);
  }
  // pre-preocess
  config_setting.ds_size_ = fSettings["ds_size"];
  config_setting.useful_corner_num_ = fSettings["useful_corner_num"];
  config_setting.stop_skip_enable_ = fSettings["stop_skip_enable"];
  config_setting.point_skip_ = fSettings["point_skip"];

  // key points
  config_setting.plane_merge_normal_thre_ =
      fSettings["plane_merge_normal_thre"];
  config_setting.plane_merge_dis_thre_ = fSettings["plane_merge_dis_thre"];
  config_setting.plane_detection_thre_ = fSettings["plane_detection_thre"];
  config_setting.voxel_size_ = fSettings["voxel_size"];
  config_setting.voxel_init_num_ = fSettings["voxel_init_num"];
  config_setting.proj_plane_num_ = fSettings["proj_plane_num"];
  config_setting.proj_image_resolution_ = fSettings["proj_image_resolution"];
  config_setting.proj_image_high_inc_ = fSettings["proj_image_high_inc"];
  config_setting.proj_dis_min_ = fSettings["proj_dis_min"];
  config_setting.proj_dis_max_ = fSettings["proj_dis_max"];
  config_setting.summary_min_thre_ = fSettings["summary_min_thre"];
  config_setting.line_filter_enable_ = fSettings["line_filter_enable"];
  config_setting.touch_filter_enable_ = fSettings["touch_filter_enable"];

  // std descriptor
  config_setting.descriptor_near_num_ = fSettings["descriptor_near_num"];
  config_setting.descriptor_min_len_ = fSettings["descriptor_min_len"];
  config_setting.descriptor_max_len_ = fSettings["descriptor_max_len"];
  config_setting.non_max_suppression_radius_ = fSettings["max_constrait_dis"];
  config_setting.std_side_resolution_ = fSettings["triangle_resolution"];

  // candidate search
  config_setting.skip_near_num_ = fSettings["skip_near_num"];
  config_setting.candidate_num_ = fSettings["candidate_num"];
  config_setting.sub_frame_num_ = fSettings["sub_frame_num"];
  config_setting.rough_dis_threshold_ = fSettings["rough_dis_threshold"];
  config_setting.similarity_threshold_ = fSettings["similarity_threshold"];
  config_setting.icp_threshold_ = fSettings["icp_threshold"];
  config_setting.ransac_Rt_thr = fSettings["ransac_Rt_thr"];

  config_setting.normal_threshold_ = fSettings["normal_threshold"];
  config_setting.dis_threshold_ = fSettings["dis_threshold"];

  // result record
  config_setting.is_kitti_ = fSettings["is_kitti"];

  // data base
  config_setting.std_add_skip_frame_ = fSettings["std_add_skip_frame"];

  // extrinsic
  cv::Mat T_lidar_to_vehicle;
  fSettings["T_lidar_to_vehicle"] >> T_lidar_to_vehicle;
  config_setting.rot_lidar_to_vehicle_ << T_lidar_to_vehicle.at<double>(0, 0),
      T_lidar_to_vehicle.at<double>(0, 1), T_lidar_to_vehicle.at<double>(0, 2),
      T_lidar_to_vehicle.at<double>(1, 0), T_lidar_to_vehicle.at<double>(1, 1),
      T_lidar_to_vehicle.at<double>(1, 2), T_lidar_to_vehicle.at<double>(2, 0),
      T_lidar_to_vehicle.at<double>(2, 1), T_lidar_to_vehicle.at<double>(2, 2);
  config_setting.t_lidar_to_vehicle_ << T_lidar_to_vehicle.at<double>(0, 3),
      T_lidar_to_vehicle.at<double>(1, 3), T_lidar_to_vehicle.at<double>(2, 3);

  config_setting.gt_file_style_ = fSettings["gt_file_style"];

  std::cout << "Sucessfully load config file:" << config_file << std::endl;
}

void load_pose_with_time(
    const std::string &pose_file,
    std::vector<std::pair<Eigen::Vector3d, Eigen::Matrix3d>> &pose_list,
    std::vector<double> &time_list) {
  time_list.clear();
  pose_list.clear();
  std::ifstream fin(pose_file);
  std::string line;
  Eigen::Matrix<double, 1, 7> temp_matrix;
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
        time_list.push_back(time);
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
          Eigen::Quaterniond q(temp_matrix[3], temp_matrix[4], temp_matrix[5],
                               temp_matrix[6]);
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

void load_cu_pose_with_time(
    const std::string &pose_file,
    std::vector<std::pair<Eigen::Vector3d, Eigen::Matrix3d>> &pose_list,
    std::vector<double> &time_list) {
  time_list.clear();
  pose_list.clear();
  std::ifstream fin(pose_file);
  std::string line;
  Eigen::Matrix<double, 1, 12> temp_matrix;
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
        time_list.push_back(time);
        number++;
      } else {
        double p;
        std::stringstream data;
        data << info;
        data >> p;
        temp_matrix[number - 1] = p;
        if (number == 12) {
          Eigen::Vector3d translation(temp_matrix[3], temp_matrix[7],
                                      temp_matrix[11]);
          Eigen::Matrix3d rotation;
          rotation << temp_matrix[0], temp_matrix[1], temp_matrix[2],
              temp_matrix[4], temp_matrix[5], temp_matrix[6], temp_matrix[8],
              temp_matrix[9], temp_matrix[10];
          std::pair<Eigen::Vector3d, Eigen::Matrix3d> single_pose;
          single_pose.first = translation;
          single_pose.second = rotation;
          pose_list.push_back(single_pose);
        }
        number++;
      }
    }
  }
}

void load_pose_with_frame(
    const std::string &pose_file,
    std::vector<std::pair<Eigen::Vector3d, Eigen::Matrix3d>> &pose_list,
    std::vector<int> &frame_number_list) {
  frame_number_list.clear();
  pose_list.clear();
  std::ifstream fin(pose_file);
  std::string line;
  Eigen::Matrix<double, 1, 12> temp_matrix;
  while (getline(fin, line)) {
    std::istringstream sin(line);
    std::vector<std::string> Waypoints;
    std::string info;
    int number = 0;
    while (getline(sin, info, ' ')) {
      if (number == 0) {
        int frame_number;
        std::stringstream data;
        data << info;
        data >> frame_number;
        frame_number_list.push_back(frame_number);
        number++;
      } else {
        double p;
        std::stringstream data;
        data << info;
        data >> p;
        temp_matrix[number - 1] = p;
        if (number == 12) {
          Eigen::Vector3d translation(temp_matrix[3], temp_matrix[7],
                                      temp_matrix[11]);
          Eigen::Matrix3d rotation;
          rotation << temp_matrix[0], temp_matrix[1], temp_matrix[2],
              temp_matrix[4], temp_matrix[5], temp_matrix[6], temp_matrix[8],
              temp_matrix[9], temp_matrix[10];
          std::pair<Eigen::Vector3d, Eigen::Matrix3d> single_pose;
          single_pose.first = translation;
          single_pose.second = rotation;
          pose_list.push_back(single_pose);
        }
        number++;
      }
    }
  }
}

void down_sampling_voxel(pcl::PointCloud<pcl::PointXYZI> &pl_feat,
                         double voxel_size) {
  int intensity = rand() % 255;
  if (voxel_size < 0.01) {
    return;
  }
  std::unordered_map<VOXEL_LOC, M_POINT> voxel_map;
  uint plsize = pl_feat.size();

  for (uint i = 0; i < plsize; i++) {
    pcl::PointXYZI &p_c = pl_feat[i];
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = p_c.data[j] / voxel_size;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }

    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                       (int64_t)loc_xyz[2]);
    auto iter = voxel_map.find(position);
    if (iter != voxel_map.end()) {
      iter->second.xyz[0] += p_c.x;
      iter->second.xyz[1] += p_c.y;
      iter->second.xyz[2] += p_c.z;
      iter->second.intensity += p_c.intensity;
      iter->second.count++;
    } else {
      M_POINT anp;
      anp.xyz[0] = p_c.x;
      anp.xyz[1] = p_c.y;
      anp.xyz[2] = p_c.z;
      anp.intensity = p_c.intensity;
      anp.count = 1;
      voxel_map[position] = anp;
    }
  }
  plsize = voxel_map.size();
  pl_feat.clear();
  pl_feat.resize(plsize);

  uint i = 0;
  for (auto iter = voxel_map.begin(); iter != voxel_map.end(); ++iter) {
    pl_feat[i].x = iter->second.xyz[0] / iter->second.count;
    pl_feat[i].y = iter->second.xyz[1] / iter->second.count;
    pl_feat[i].z = iter->second.xyz[2] / iter->second.count;
    pl_feat[i].intensity = iter->second.intensity / iter->second.count;
    i++;
  }
}
double binary_similarity(const BinaryDescriptor &b1,
                         const BinaryDescriptor &b2) {
  double dis = 0;
  for (size_t i = 0; i < b1.occupy_array_.size(); i++) {
    // to be debug hanming distance
    if (b1.occupy_array_[i] == true && b2.occupy_array_[i] == true) {
      dis += 1;
    }
  }
  return 2 * dis / (b1.summary_ + b2.summary_);
}

bool binary_greater_sort(BinaryDescriptor a, BinaryDescriptor b) {
  return (a.summary_ > b.summary_);
}

bool plane_greater_sort(Plane *plane1, Plane *plane2) {
  return plane1->sub_plane_num_ > plane2->sub_plane_num_;
}

void init_voxel_map(const ConfigSetting &config_setting,
                    const pcl::PointCloud<pcl::PointXYZI> &input_cloud,
                    std::unordered_map<VOXEL_LOC, OctoTree *> &voxel_map) {
  uint plsize = input_cloud.size();
  for (uint i = 0; i < plsize; i++) {
    Eigen::Vector3d p_c(input_cloud[i].x, input_cloud[i].y, input_cloud[i].z);
    double loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = p_c[j] / config_setting.voxel_size_;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }
    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                       (int64_t)loc_xyz[2]);
    auto iter = voxel_map.find(position);
    if (iter != voxel_map.end()) {
      voxel_map[position]->voxel_points_.push_back(p_c);
    } else {
      OctoTree *octo_tree = new OctoTree(config_setting);
      voxel_map[position] = octo_tree;
      voxel_map[position]->voxel_points_.push_back(p_c);
    }
  }
  std::vector<std::unordered_map<VOXEL_LOC, OctoTree *>::iterator> iter_list;
  std::vector<size_t> index;
  size_t i = 0;
  for (auto iter = voxel_map.begin(); iter != voxel_map.end(); ++iter) {
    index.push_back(i);
    i++;
    iter_list.push_back(iter);
    // iter->second->init_octo_tree();
  }
  std::for_each(
      std::execution::par_unseq, index.begin(), index.end(),
      [&](const size_t &i) { iter_list[i]->second->init_octo_tree(); });
}

void OctoTree::init_octo_tree() {
  if (voxel_points_.size() > config_setting_.voxel_init_num_) {
    init_plane();
  }
}

void OctoTree::init_plane() {
  plane_ptr_->covariance_ = Eigen::Matrix3d::Zero();
  plane_ptr_->center_ = Eigen::Vector3d::Zero();
  plane_ptr_->normal_ = Eigen::Vector3d::Zero();
  plane_ptr_->points_size_ = voxel_points_.size();
  plane_ptr_->radius_ = 0;
  for (auto pi : voxel_points_) {
    plane_ptr_->covariance_ += pi * pi.transpose();
    plane_ptr_->center_ += pi;
  }
  plane_ptr_->center_ = plane_ptr_->center_ / plane_ptr_->points_size_;
  plane_ptr_->covariance_ =
      plane_ptr_->covariance_ / plane_ptr_->points_size_ -
      plane_ptr_->center_ * plane_ptr_->center_.transpose();
  Eigen::EigenSolver<Eigen::Matrix3d> es(plane_ptr_->covariance_);
  Eigen::Matrix3cd evecs = es.eigenvectors();
  Eigen::Vector3cd evals = es.eigenvalues();
  Eigen::Vector3d evalsReal;
  evalsReal = evals.real();
  Eigen::Matrix3d::Index evalsMin, evalsMax;
  evalsReal.rowwise().sum().minCoeff(&evalsMin);
  evalsReal.rowwise().sum().maxCoeff(&evalsMax);
  int evalsMid = 3 - evalsMin - evalsMax;
  if (evalsReal(evalsMin) < config_setting_.plane_detection_thre_) {
    plane_ptr_->normal_ << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin),
        evecs.real()(2, evalsMin);
    plane_ptr_->min_eigen_value_ = evalsReal(evalsMin);
    plane_ptr_->radius_ = sqrt(evalsReal(evalsMax));
    plane_ptr_->is_plane_ = true;

    plane_ptr_->d_ = -(plane_ptr_->normal_(0) * plane_ptr_->center_(0) +
                       plane_ptr_->normal_(1) * plane_ptr_->center_(1) +
                       plane_ptr_->normal_(2) * plane_ptr_->center_(2));
    plane_ptr_->p_center_.x = plane_ptr_->center_(0);
    plane_ptr_->p_center_.y = plane_ptr_->center_(1);
    plane_ptr_->p_center_.z = plane_ptr_->center_(2);
    plane_ptr_->p_center_.normal_x = plane_ptr_->normal_(0);
    plane_ptr_->p_center_.normal_y = plane_ptr_->normal_(1);
    plane_ptr_->p_center_.normal_z = plane_ptr_->normal_(2);
  } else {
    plane_ptr_->is_plane_ = false;
  }
}

void get_plane(std::unordered_map<VOXEL_LOC, OctoTree *> &voxel_map,
               pcl::PointCloud<pcl::PointXYZINormal>::Ptr &plane_cloud) {
  for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++) {
    if (iter->second->plane_ptr_->is_plane_) {
      pcl::PointXYZINormal pi;
      pi.x = iter->second->plane_ptr_->center_[0];
      pi.y = iter->second->plane_ptr_->center_[1];
      pi.z = iter->second->plane_ptr_->center_[2];
      pi.normal_x = iter->second->plane_ptr_->normal_[0];
      pi.normal_y = iter->second->plane_ptr_->normal_[1];
      pi.normal_z = iter->second->plane_ptr_->normal_[2];
      plane_cloud->push_back(pi);
    }
  }
}

void get_project_plane(
    const ConfigSetting &config_setting,
    std::unordered_map<VOXEL_LOC, OctoTree *> &descriptor_map,

    std::vector<Plane *> &project_plane_list) {
  std::vector<Plane *> origin_list;
  for (auto iter = descriptor_map.begin(); iter != descriptor_map.end();
       iter++) {
    if (iter->second->plane_ptr_->is_plane_) {
      origin_list.push_back(iter->second->plane_ptr_);
    }
  }
  for (size_t i = 0; i < origin_list.size(); i++)
    origin_list[i]->id_ = 0;
  int current_id = 1;
  for (auto iter = origin_list.end() - 1; iter != origin_list.begin(); iter--) {
    for (auto iter2 = origin_list.begin(); iter2 != iter; iter2++) {
      Eigen::Vector3d normal_diff = (*iter)->normal_ - (*iter2)->normal_;
      Eigen::Vector3d normal_add = (*iter)->normal_ + (*iter2)->normal_;
      double dis1 =
          fabs((*iter)->normal_(0) * (*iter2)->center_(0) +
               (*iter)->normal_(1) * (*iter2)->center_(1) +
               (*iter)->normal_(2) * (*iter2)->center_(2) + (*iter)->d_);
      double dis2 =
          fabs((*iter2)->normal_(0) * (*iter)->center_(0) +
               (*iter2)->normal_(1) * (*iter)->center_(1) +
               (*iter2)->normal_(2) * (*iter)->center_(2) + (*iter2)->d_);
      if (normal_diff.norm() < config_setting.plane_merge_normal_thre_ ||
          normal_add.norm() < config_setting.plane_merge_normal_thre_)
        if (dis1 < config_setting.plane_merge_dis_thre_ &&
            dis2 < config_setting.plane_merge_dis_thre_) {
          if ((*iter)->id_ == 0 && (*iter2)->id_ == 0) {
            (*iter)->id_ = current_id;
            (*iter2)->id_ = current_id;
            current_id++;
          } else if ((*iter)->id_ == 0 && (*iter2)->id_ != 0)
            (*iter)->id_ = (*iter2)->id_;
          else if ((*iter)->id_ != 0 && (*iter2)->id_ == 0)
            (*iter2)->id_ = (*iter)->id_;
        }
    }
  }
  std::vector<Plane *> merge_list;
  std::vector<int> merge_flag;

  for (size_t i = 0; i < origin_list.size(); i++) {
    auto it =
        std::find(merge_flag.begin(), merge_flag.end(), origin_list[i]->id_);
    if (it != merge_flag.end())
      continue;
    if (origin_list[i]->id_ == 0) {
      continue;
    }
    Plane *merge_plane = new Plane;
    (*merge_plane) = (*origin_list[i]);
    bool is_merge = false;
    for (size_t j = 0; j < origin_list.size(); j++) {
      if (i == j)
        continue;
      if (origin_list[j]->id_ == origin_list[i]->id_) {
        is_merge = true;
        Eigen::Matrix3d P_PT1 =
            (merge_plane->covariance_ +
             merge_plane->center_ * merge_plane->center_.transpose()) *
            merge_plane->points_size_;
        Eigen::Matrix3d P_PT2 =
            (origin_list[j]->covariance_ +
             origin_list[j]->center_ * origin_list[j]->center_.transpose()) *
            origin_list[j]->points_size_;
        Eigen::Vector3d merge_center =
            (merge_plane->center_ * merge_plane->points_size_ +
             origin_list[j]->center_ * origin_list[j]->points_size_) /
            (merge_plane->points_size_ + origin_list[j]->points_size_);
        Eigen::Matrix3d merge_covariance =
            (P_PT1 + P_PT2) /
                (merge_plane->points_size_ + origin_list[j]->points_size_) -
            merge_center * merge_center.transpose();
        merge_plane->covariance_ = merge_covariance;
        merge_plane->center_ = merge_center;
        merge_plane->points_size_ =
            merge_plane->points_size_ + origin_list[j]->points_size_;
        merge_plane->sub_plane_num_++;
        // for (size_t k = 0; k < origin_list[j]->cloud.size(); k++) {
        //   merge_plane->cloud.points.push_back(origin_list[j]->cloud.points[k]);
        // }
        Eigen::EigenSolver<Eigen::Matrix3d> es(merge_plane->covariance_);
        Eigen::Matrix3cd evecs = es.eigenvectors();
        Eigen::Vector3cd evals = es.eigenvalues();
        Eigen::Vector3d evalsReal;
        evalsReal = evals.real();
        Eigen::Matrix3f::Index evalsMin, evalsMax;
        evalsReal.rowwise().sum().minCoeff(&evalsMin);
        evalsReal.rowwise().sum().maxCoeff(&evalsMax);
        Eigen::Vector3d evecMin = evecs.real().col(evalsMin);
        merge_plane->normal_ << evecs.real()(0, evalsMin),
            evecs.real()(1, evalsMin), evecs.real()(2, evalsMin);
        merge_plane->radius_ = sqrt(evalsReal(evalsMax));
        merge_plane->d_ = -(merge_plane->normal_(0) * merge_plane->center_(0) +
                            merge_plane->normal_(1) * merge_plane->center_(1) +
                            merge_plane->normal_(2) * merge_plane->center_(2));
        merge_plane->p_center_.x = merge_plane->center_(0);
        merge_plane->p_center_.y = merge_plane->center_(1);
        merge_plane->p_center_.z = merge_plane->center_(2);
        merge_plane->p_center_.normal_x = merge_plane->normal_(0);
        merge_plane->p_center_.normal_y = merge_plane->normal_(1);
        merge_plane->p_center_.normal_z = merge_plane->normal_(2);
      }
    }
    if (is_merge) {
      merge_flag.push_back(merge_plane->id_);
      merge_list.push_back(merge_plane);
    }
  }
  project_plane_list = merge_list;
}

void merge_plane(const ConfigSetting &config_setting,
                 std::vector<Plane *> &origin_list,
                 std::vector<Plane *> &merge_plane_list) {
  if (origin_list.size() == 1) {
    merge_plane_list = origin_list;
    return;
  }
  for (size_t i = 0; i < origin_list.size(); i++)
    origin_list[i]->id_ = 0;
  int current_id = 1;
  for (auto iter = origin_list.end() - 1; iter != origin_list.begin(); iter--) {
    for (auto iter2 = origin_list.begin(); iter2 != iter; iter2++) {
      Eigen::Vector3d normal_diff = (*iter)->normal_ - (*iter2)->normal_;
      Eigen::Vector3d normal_add = (*iter)->normal_ + (*iter2)->normal_;
      double dis1 =
          fabs((*iter)->normal_(0) * (*iter2)->center_(0) +
               (*iter)->normal_(1) * (*iter2)->center_(1) +
               (*iter)->normal_(2) * (*iter2)->center_(2) + (*iter)->d_);
      double dis2 =
          fabs((*iter2)->normal_(0) * (*iter)->center_(0) +
               (*iter2)->normal_(1) * (*iter)->center_(1) +
               (*iter2)->normal_(2) * (*iter)->center_(2) + (*iter2)->d_);
      if (normal_diff.norm() < config_setting.plane_merge_normal_thre_ ||
          normal_add.norm() < config_setting.plane_merge_normal_thre_)
        if (dis1 < config_setting.plane_merge_dis_thre_ &&
            dis2 < config_setting.plane_merge_dis_thre_) {
          if ((*iter)->id_ == 0 && (*iter2)->id_ == 0) {
            (*iter)->id_ = current_id;
            (*iter2)->id_ = current_id;
            current_id++;
          } else if ((*iter)->id_ == 0 && (*iter2)->id_ != 0)
            (*iter)->id_ = (*iter2)->id_;
          else if ((*iter)->id_ != 0 && (*iter2)->id_ == 0)
            (*iter2)->id_ = (*iter)->id_;
        }
    }
  }
  std::vector<int> merge_flag;

  for (size_t i = 0; i < origin_list.size(); i++) {
    auto it =
        std::find(merge_flag.begin(), merge_flag.end(), origin_list[i]->id_);
    if (it != merge_flag.end())
      continue;
    if (origin_list[i]->id_ == 0) {
      merge_plane_list.push_back(origin_list[i]);
      continue;
    }
    Plane *merge_plane = new Plane;
    (*merge_plane) = (*origin_list[i]);
    bool is_merge = false;
    for (size_t j = 0; j < origin_list.size(); j++) {
      if (i == j)
        continue;
      if (origin_list[j]->id_ == origin_list[i]->id_) {
        is_merge = true;
        Eigen::Matrix3d P_PT1 =
            (merge_plane->covariance_ +
             merge_plane->center_ * merge_plane->center_.transpose()) *
            merge_plane->points_size_;
        Eigen::Matrix3d P_PT2 =
            (origin_list[j]->covariance_ +
             origin_list[j]->center_ * origin_list[j]->center_.transpose()) *
            origin_list[j]->points_size_;
        Eigen::Vector3d merge_center =
            (merge_plane->center_ * merge_plane->points_size_ +
             origin_list[j]->center_ * origin_list[j]->points_size_) /
            (merge_plane->points_size_ + origin_list[j]->points_size_);
        Eigen::Matrix3d merge_covariance =
            (P_PT1 + P_PT2) /
                (merge_plane->points_size_ + origin_list[j]->points_size_) -
            merge_center * merge_center.transpose();
        merge_plane->covariance_ = merge_covariance;
        merge_plane->center_ = merge_center;
        merge_plane->points_size_ =
            merge_plane->points_size_ + origin_list[j]->points_size_;
        merge_plane->sub_plane_num_ += origin_list[j]->sub_plane_num_;
        // for (size_t k = 0; k < origin_list[j]->cloud.size(); k++) {
        //   merge_plane->cloud.points.push_back(origin_list[j]->cloud.points[k]);
        // }
        Eigen::EigenSolver<Eigen::Matrix3d> es(merge_plane->covariance_);
        Eigen::Matrix3cd evecs = es.eigenvectors();
        Eigen::Vector3cd evals = es.eigenvalues();
        Eigen::Vector3d evalsReal;
        evalsReal = evals.real();
        Eigen::Matrix3f::Index evalsMin, evalsMax;
        evalsReal.rowwise().sum().minCoeff(&evalsMin);
        evalsReal.rowwise().sum().maxCoeff(&evalsMax);
        Eigen::Vector3d evecMin = evecs.real().col(evalsMin);
        merge_plane->normal_ << evecs.real()(0, evalsMin),
            evecs.real()(1, evalsMin), evecs.real()(2, evalsMin);
        merge_plane->radius_ = sqrt(evalsReal(evalsMax));
        merge_plane->d_ = -(merge_plane->normal_(0) * merge_plane->center_(0) +
                            merge_plane->normal_(1) * merge_plane->center_(1) +
                            merge_plane->normal_(2) * merge_plane->center_(2));
        merge_plane->p_center_.x = merge_plane->center_(0);
        merge_plane->p_center_.y = merge_plane->center_(1);
        merge_plane->p_center_.z = merge_plane->center_(2);
        merge_plane->p_center_.normal_x = merge_plane->normal_(0);
        merge_plane->p_center_.normal_y = merge_plane->normal_(1);
        merge_plane->p_center_.normal_z = merge_plane->normal_(2);
      }
    }
    if (is_merge) {
      merge_flag.push_back(merge_plane->id_);
      merge_plane_list.push_back(merge_plane);
    }
  }
}

void non_max_suppression(const ConfigSetting &config_setting,
                         std::vector<BinaryDescriptor> &binary_list) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr prepare_key_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  pcl::KdTreeFLANN<pcl::PointXYZ> kd_tree;
  std::vector<int> pre_count_list;
  std::vector<bool> is_add_list;
  for (auto var : binary_list) {
    pcl::PointXYZ pi;
    pi.x = var.location_[0];
    pi.y = var.location_[1];
    pi.z = var.location_[2];
    prepare_key_cloud->push_back(pi);
    pre_count_list.push_back(var.summary_);
    is_add_list.push_back(true);
  }
  kd_tree.setInputCloud(prepare_key_cloud);
  std::vector<int> pointIdxRadiusSearch;
  std::vector<float> pointRadiusSquaredDistance;
  double radius = config_setting.non_max_suppression_radius_;
  for (size_t i = 0; i < prepare_key_cloud->size(); i++) {
    pcl::PointXYZ searchPoint = prepare_key_cloud->points[i];
    if (kd_tree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch,
                             pointRadiusSquaredDistance) > 0) {
      Eigen::Vector3d pi(searchPoint.x, searchPoint.y, searchPoint.z);
      for (size_t j = 0; j < pointIdxRadiusSearch.size(); ++j) {
        Eigen::Vector3d pj(
            prepare_key_cloud->points[pointIdxRadiusSearch[j]].x,
            prepare_key_cloud->points[pointIdxRadiusSearch[j]].y,
            prepare_key_cloud->points[pointIdxRadiusSearch[j]].z);
        if (pointIdxRadiusSearch[j] == i) {
          continue;
        }
        if (pre_count_list[i] <= pre_count_list[pointIdxRadiusSearch[j]]) {
          is_add_list[i] = false;
        }
      }
    }
  }
  std::vector<BinaryDescriptor> pass_binary_list;
  for (size_t i = 0; i < is_add_list.size(); i++) {
    if (is_add_list[i]) {
      pass_binary_list.push_back(binary_list[i]);
    }
  }
  binary_list.clear();
  for (auto var : pass_binary_list) {
    binary_list.push_back(var);
  }
  return;
}

void binary_extractor(const ConfigSetting &config_setting,
                      const std::vector<Plane *> proj_plane_list,
                      const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
                      std::vector<BinaryDescriptor> &binary_descriptor_list) {
  binary_descriptor_list.clear();
  std::vector<BinaryDescriptor> temp_binary_list;
  Eigen::Vector3d last_normal(0, 0, 0);
  int useful_proj_num = 0;
  for (int i = 0; i < proj_plane_list.size(); i++) {
    std::vector<BinaryDescriptor> prepare_binary_list;
    Eigen::Vector3d proj_center = proj_plane_list[i]->center_;
    Eigen::Vector3d proj_normal = proj_plane_list[i]->normal_;
    if ((proj_normal - last_normal).norm() < 0.3 ||
        (proj_normal + last_normal).norm() > 0.3) {
      last_normal = proj_normal;
      std::cout << "proj normal:" << proj_normal.transpose() << std::endl;
      useful_proj_num++;
      extract_binary(config_setting, proj_center, proj_normal, input_cloud,
                     prepare_binary_list);
      for (auto bi : prepare_binary_list) {
        temp_binary_list.push_back(bi);
      }
      if (useful_proj_num == config_setting.proj_plane_num_) {
        break;
      }
    }
  }

  // for (size_t i = 0; i < config_setting.proj_plane_num_; i++) {
  //   if (i >= proj_plane_list.size()) {
  //     break;
  //   }
  //   std::vector<BinaryDescriptor> prepare_binary_list;
  //   Eigen::Vector3d proj_center = proj_plane_list[i]->center_;
  //   Eigen::Vector3d proj_normal = proj_plane_list[i]->normal_;
  //   extract_binary(config_setting, proj_center, proj_normal, input_cloud,
  //                  prepare_binary_list);
  //   for (auto bi : prepare_binary_list) {
  //     temp_binary_list.push_back(bi);
  //   }
  // }
  non_max_suppression(config_setting, temp_binary_list);
  if (config_setting.useful_corner_num_ > temp_binary_list.size()) {
    binary_descriptor_list = temp_binary_list;
  } else {
    std::sort(temp_binary_list.begin(), temp_binary_list.end(),
              binary_greater_sort);
    for (size_t i = 0; i < config_setting.useful_corner_num_; i++) {
      binary_descriptor_list.push_back(temp_binary_list[i]);
    }
  }
  return;
}

void binary_extractor_debug(
    const ConfigSetting &config_setting,
    const std::vector<Plane *> proj_plane_list,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
    std::vector<BinaryDescriptor> &binary_descriptor_list,
    std::vector<BinaryDescriptor> &binary_descriptor_around_list) {
  binary_descriptor_list.clear();
  binary_descriptor_around_list.clear();
  std::vector<BinaryDescriptor> temp_binary_list;
  std::vector<BinaryDescriptor> temp_binary_around_list;
  for (size_t i = 0; i < config_setting.proj_plane_num_; i++) {
    std::vector<BinaryDescriptor> prepare_binary_list;
    std::vector<BinaryDescriptor> prepare_binary_around_list;
    Eigen::Vector3d proj_center = proj_plane_list[i]->center_;
    Eigen::Vector3d proj_normal = proj_plane_list[i]->normal_;
    extract_binary_debug(config_setting, proj_center, proj_normal, input_cloud,
                         prepare_binary_list, prepare_binary_around_list);
    for (auto bi : prepare_binary_list) {
      binary_descriptor_list.push_back(bi);
    }
    for (auto bi : prepare_binary_around_list) {
      binary_descriptor_around_list.push_back(bi);
    }
  }
  // non_max_suppression(config_setting, temp_binary_list);
  // if (config_setting.useful_corner_num_ > temp_binary_list.size()) {
  //   binary_descriptor_list = temp_binary_list;
  // } else {
  //   std::sort(temp_binary_list.begin(), temp_binary_list.end(),
  //             binary_greater_sort);
  //   for (size_t i = 0; i < config_setting.useful_corner_num_; i++) {
  //     binary_descriptor_list.push_back(temp_binary_list[i]);
  //   }
  // }
  return;
}

void extract_binary(const ConfigSetting &config_setting,
                    const Eigen::Vector3d &project_center,
                    const Eigen::Vector3d &project_normal,
                    const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
                    std::vector<BinaryDescriptor> &binary_list) {
  binary_list.clear();
  double binary_min_dis = config_setting.summary_min_thre_;
  double resolution = config_setting.proj_image_resolution_;
  double dis_threshold_min = config_setting.proj_dis_min_;
  double dis_threshold_max = config_setting.proj_dis_max_;
  double high_inc = config_setting.proj_image_high_inc_;
  bool line_filter_enable = config_setting.line_filter_enable_;
  double A = project_normal[0];
  double B = project_normal[1];
  double C = project_normal[2];
  double D =
      -(A * project_center[0] + B * project_center[1] + C * project_center[2]);
  std::vector<Eigen::Vector3d> projection_points;
  Eigen::Vector3d x_axis(1, 1, 0);
  if (C != 0) {
    x_axis[2] = -(A + B) / C;
  } else if (B != 0) {
    x_axis[1] = -A / B;
  } else {
    x_axis[0] = 0;
    x_axis[1] = 1;
  }
  x_axis.normalize();
  Eigen::Vector3d y_axis = project_normal.cross(x_axis);
  y_axis.normalize();
  double ax = x_axis[0];
  double bx = x_axis[1];
  double cx = x_axis[2];
  double dx = -(ax * project_center[0] + bx * project_center[1] +
                cx * project_center[2]);
  double ay = y_axis[0];
  double by = y_axis[1];
  double cy = y_axis[2];
  double dy = -(ay * project_center[0] + by * project_center[1] +
                cy * project_center[2]);
  std::vector<Eigen::Vector2d> point_list_2d;
  pcl::PointCloud<pcl::PointXYZ> point_list_3d;
  std::vector<double> dis_list_2d;
  for (size_t i = 0; i < input_cloud->size(); i++) {
    double x = input_cloud->points[i].x;
    double y = input_cloud->points[i].y;
    double z = input_cloud->points[i].z;
    double dis = fabs(x * A + y * B + z * C + D);
    pcl::PointXYZ pi;
    if (dis < dis_threshold_min || dis > dis_threshold_max) {
      continue;
    } else {
      if (dis > dis_threshold_min && dis <= dis_threshold_max) {
        pi.x = x;
        pi.y = y;
        pi.z = z;
      }
    }
    Eigen::Vector3d cur_project;

    cur_project[0] = (-A * (B * y + C * z + D) + x * (B * B + C * C)) /
                     (A * A + B * B + C * C);
    cur_project[1] = (-B * (A * x + C * z + D) + y * (A * A + C * C)) /
                     (A * A + B * B + C * C);
    cur_project[2] = (-C * (A * x + B * y + D) + z * (A * A + B * B)) /
                     (A * A + B * B + C * C);
    pcl::PointXYZ p;
    p.x = cur_project[0];
    p.y = cur_project[1];
    p.z = cur_project[2];
    double project_x =
        cur_project[0] * ay + cur_project[1] * by + cur_project[2] * cy + dy;
    double project_y =
        cur_project[0] * ax + cur_project[1] * bx + cur_project[2] * cx + dx;
    Eigen::Vector2d p_2d(project_x, project_y);
    point_list_2d.push_back(p_2d);
    dis_list_2d.push_back(dis);
    point_list_3d.points.push_back(pi);
  }
  double min_x = 10;
  double max_x = -10;
  double min_y = 10;
  double max_y = -10;
  if (point_list_2d.size() <= 5) {
    return;
  }
  for (auto pi : point_list_2d) {
    if (pi[0] < min_x) {
      min_x = pi[0];
    }
    if (pi[0] > max_x) {
      max_x = pi[0];
    }
    if (pi[1] < min_y) {
      min_y = pi[1];
    }
    if (pi[1] > max_y) {
      max_y = pi[1];
    }
  }
  // segment project cloud
  int segmen_base_num = 5;
  double segmen_len = segmen_base_num * resolution;
  int x_segment_num = (max_x - min_x) / segmen_len + 1;
  int y_segment_num = (max_y - min_y) / segmen_len + 1;
  int x_axis_len = (int)((max_x - min_x) / resolution + segmen_base_num);
  int y_axis_len = (int)((max_y - min_y) / resolution + segmen_base_num);

  std::vector<double> **dis_container = new std::vector<double> *[x_axis_len];
  BinaryDescriptor **binary_container = new BinaryDescriptor *[x_axis_len];
  for (int i = 0; i < x_axis_len; i++) {
    dis_container[i] = new std::vector<double>[y_axis_len];
    binary_container[i] = new BinaryDescriptor[y_axis_len];
  }
  double **img_count = new double *[x_axis_len];
  for (int i = 0; i < x_axis_len; i++) {
    img_count[i] = new double[y_axis_len];
  }
  double **dis_array = new double *[x_axis_len];
  for (int i = 0; i < x_axis_len; i++) {
    dis_array[i] = new double[y_axis_len];
  }
  double **mean_x_list = new double *[x_axis_len];
  for (int i = 0; i < x_axis_len; i++) {
    mean_x_list[i] = new double[y_axis_len];
  }
  double **mean_y_list = new double *[x_axis_len];
  for (int i = 0; i < x_axis_len; i++) {
    mean_y_list[i] = new double[y_axis_len];
  }
  for (int x = 0; x < x_axis_len; x++) {
    for (int y = 0; y < y_axis_len; y++) {
      img_count[x][y] = 0;
      mean_x_list[x][y] = 0;
      mean_y_list[x][y] = 0;
      dis_array[x][y] = 0;
      std::vector<double> single_dis_container;
      dis_container[x][y] = single_dis_container;
    }
  }

  for (size_t i = 0; i < point_list_2d.size(); i++) {
    int x_index = (int)((point_list_2d[i][0] - min_x) / resolution);
    int y_index = (int)((point_list_2d[i][1] - min_y) / resolution);
    mean_x_list[x_index][y_index] += point_list_2d[i][0];
    mean_y_list[x_index][y_index] += point_list_2d[i][1];
    img_count[x_index][y_index]++;
    dis_container[x_index][y_index].push_back(dis_list_2d[i]);
  }

  for (int x = 0; x < x_axis_len; x++) {
    for (int y = 0; y < y_axis_len; y++) {
      // calc segment dis array
      if (img_count[x][y] > 0) {
        int cut_num = (dis_threshold_max - dis_threshold_min) / high_inc;
        std::vector<bool> occup_list;
        std::vector<double> cnt_list;
        BinaryDescriptor single_binary;
        for (size_t i = 0; i < cut_num; i++) {
          cnt_list.push_back(0);
          occup_list.push_back(false);
        }
        for (size_t j = 0; j < dis_container[x][y].size(); j++) {
          int cnt_index =
              (dis_container[x][y][j] - dis_threshold_min) / high_inc;
          cnt_list[cnt_index]++;
        }
        double segmnt_dis = 0;
        for (size_t i = 0; i < cut_num; i++) {
          if (cnt_list[i] >= 1) {
            segmnt_dis++;
            occup_list[i] = true;
          }
        }
        dis_array[x][y] = segmnt_dis;
        single_binary.occupy_array_ = occup_list;
        single_binary.summary_ = segmnt_dis;
        binary_container[x][y] = single_binary;
      }
    }
  }

  // debug image
  // double max_dis_cnt = (dis_threshold_max - dis_threshold_min) / high_inc;
  // cv::Mat proj_image = cv::Mat::zeros(y_axis_len, x_axis_len, CV_8UC1);
  // for (size_t y = 0; y < y_axis_len; y++) {
  //   for (size_t x = 0; x < x_axis_len; x++) {
  //     if (dis_array[x][y] != 0) {
  //       proj_image.at<uchar>(y, x) = dis_array[x][y] * 20 + 50;
  //     }
  //   }
  // }
  // cv::Mat image_max; // 等比例放大图
  // cv::resize(proj_image, image_max,
  //            cv::Size(y_axis_len * 2, x_axis_len * 2)); // 放大操作
  // cv::Mat out;
  // // cv::equalizeHist(proj_image, out);
  // cv::imshow("proj image", proj_image);
  // cv::waitKey();

  // filter by distance
  std::vector<double> max_dis_list;
  std::vector<int> max_dis_x_index_list;
  std::vector<int> max_dis_y_index_list;

  for (int x_segment_index = 0; x_segment_index < x_segment_num;
       x_segment_index++) {
    for (int y_segment_index = 0; y_segment_index < y_segment_num;
         y_segment_index++) {
      double max_dis = 0;
      int max_dis_x_index = -10;
      int max_dis_y_index = -10;
      for (int x_index = x_segment_index * segmen_base_num;
           x_index < (x_segment_index + 1) * segmen_base_num; x_index++) {
        for (int y_index = y_segment_index * segmen_base_num;
             y_index < (y_segment_index + 1) * segmen_base_num; y_index++) {
          if (dis_array[x_index][y_index] > max_dis) {
            max_dis = dis_array[x_index][y_index];
            max_dis_x_index = x_index;
            max_dis_y_index = y_index;
          }
        }
      }
      if (max_dis >= binary_min_dis) {
        bool is_touch = true;
        if (config_setting.touch_filter_enable_) {
          is_touch = binary_container[max_dis_x_index][max_dis_y_index]
                         .occupy_array_[0] ||
                     binary_container[max_dis_x_index][max_dis_y_index]
                         .occupy_array_[1] ||
                     binary_container[max_dis_x_index][max_dis_y_index]
                         .occupy_array_[2] ||
                     binary_container[max_dis_x_index][max_dis_y_index]
                         .occupy_array_[3];
        }

        if (is_touch) {
          max_dis_list.push_back(max_dis);
          max_dis_x_index_list.push_back(max_dis_x_index);
          max_dis_y_index_list.push_back(max_dis_y_index);
        }
      }
    }
  }
  // calc line or not
  std::vector<Eigen::Vector2i> direction_list;
  Eigen::Vector2i d(0, 1);
  direction_list.push_back(d);
  d << 1, 0;
  direction_list.push_back(d);
  d << 1, 1;
  direction_list.push_back(d);
  d << 1, -1;
  direction_list.push_back(d);
  for (size_t i = 0; i < max_dis_list.size(); i++) {
    Eigen::Vector2i p(max_dis_x_index_list[i], max_dis_y_index_list[i]);
    if (p[0] <= 0 || p[0] >= x_axis_len - 1 || p[1] <= 0 ||
        p[1] >= y_axis_len - 1) {
      continue;
    }
    bool is_add = true;
    // if (line_filter_enable) {
    //   for (int dx = -1; dx <= 1; dx++) {
    //     for (int dy = -1; dy <= 1; dy++) {
    //       if (dx == 0 && dy == 0) {
    //         continue;
    //       }
    //       Eigen::Vector2i p_near = p;
    //       p_near[0] = p[0] + dx;
    //       p_near[1] = p[1] + dy;
    //       double threshold = dis_array[p[0]][p[1]] - 3;
    //       if (dis_array[p_near[0]][p_near[1]] >= threshold) {
    //         is_add = false;
    //       }
    //     }
    //   }
    // }

    if (line_filter_enable) {
      for (int j = 0; j < 4; j++) {
        Eigen::Vector2i p(max_dis_x_index_list[i], max_dis_y_index_list[i]);
        if (p[0] <= 0 || p[0] >= x_axis_len - 1 || p[1] <= 0 ||
            p[1] >= y_axis_len - 1) {
          continue;
        }
        Eigen::Vector2i p1 = p + direction_list[j];
        Eigen::Vector2i p2 = p - direction_list[j];
        double threshold = dis_array[p[0]][p[1]] - 3;
        if (dis_array[p1[0]][p1[1]] >= threshold) {
          if (dis_array[p2[0]][p2[1]] >= 0.5 * dis_array[p[0]][p[1]]) {
            is_add = false;
          }
        }
        if (dis_array[p2[0]][p2[1]] >= threshold) {
          if (dis_array[p1[0]][p1[1]] >= 0.5 * dis_array[p[0]][p[1]]) {
            is_add = false;
          }
        }
        if (dis_array[p1[0]][p1[1]] >= threshold) {
          if (dis_array[p2[0]][p2[1]] >= threshold) {
            is_add = false;
          }
        }
        if (dis_array[p2[0]][p2[1]] >= threshold) {
          if (dis_array[p1[0]][p1[1]] >= threshold) {
            is_add = false;
          }
        }
      }
    }
    if (is_add) {
      double px =
          mean_x_list[max_dis_x_index_list[i]][max_dis_y_index_list[i]] /
          img_count[max_dis_x_index_list[i]][max_dis_y_index_list[i]];
      double py =
          mean_y_list[max_dis_x_index_list[i]][max_dis_y_index_list[i]] /
          img_count[max_dis_x_index_list[i]][max_dis_y_index_list[i]];
      Eigen::Vector3d coord = py * x_axis + px * y_axis + project_center;
      pcl::PointXYZ pi;
      pi.x = coord[0];
      pi.y = coord[1];
      pi.z = coord[2];
      BinaryDescriptor single_binary =
          binary_container[max_dis_x_index_list[i]][max_dis_y_index_list[i]];
      single_binary.location_ = coord;
      binary_list.push_back(single_binary);
    }
  }
  for (int i = 0; i < x_axis_len; i++) {
    delete[] binary_container[i];
    delete[] dis_container[i];
    delete[] img_count[i];
    delete[] dis_array[i];
    delete[] mean_x_list[i];
    delete[] mean_y_list[i];
  }
  delete[] binary_container;
  delete[] dis_container;
  delete[] img_count;
  delete[] dis_array;
  delete[] mean_x_list;
  delete[] mean_y_list;
}

void extract_binary_debug(
    const ConfigSetting &config_setting, const Eigen::Vector3d &project_center,
    const Eigen::Vector3d &project_normal,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
    std::vector<BinaryDescriptor> &binary_list,
    std::vector<BinaryDescriptor> &binary_around_list) {
  binary_list.clear();
  double binary_min_dis = config_setting.summary_min_thre_;
  double resolution = config_setting.proj_image_resolution_;
  double dis_threshold_min = config_setting.proj_dis_min_;
  double dis_threshold_max = config_setting.proj_dis_max_;
  double high_inc = config_setting.proj_image_high_inc_;
  bool line_filter_enable = config_setting.line_filter_enable_;
  double A = project_normal[0];
  double B = project_normal[1];
  double C = project_normal[2];
  double D =
      -(A * project_center[0] + B * project_center[1] + C * project_center[2]);
  std::vector<Eigen::Vector3d> projection_points;
  Eigen::Vector3d x_axis(1, 100, 0);
  if (C != 0) {
    x_axis[2] = -(A + B) / C;
  } else if (B != 0) {
    x_axis[1] = -A / B;
  } else {
    x_axis[0] = 0;
    x_axis[1] = 1;
  }
  x_axis.normalize();
  Eigen::Vector3d y_axis = project_normal.cross(x_axis);
  y_axis.normalize();
  double ax = x_axis[0];
  double bx = x_axis[1];
  double cx = x_axis[2];
  double dx = -(ax * project_center[0] + bx * project_center[1] +
                cx * project_center[2]);
  double ay = y_axis[0];
  double by = y_axis[1];
  double cy = y_axis[2];
  double dy = -(ay * project_center[0] + by * project_center[1] +
                cy * project_center[2]);
  std::vector<Eigen::Vector2d> point_list_2d;
  pcl::PointCloud<pcl::PointXYZ> point_list_3d;
  std::vector<double> dis_list_2d;
  for (size_t i = 0; i < input_cloud->size(); i++) {
    double x = input_cloud->points[i].x;
    double y = input_cloud->points[i].y;
    double z = input_cloud->points[i].z;
    double dis = fabs(x * A + y * B + z * C + D);
    pcl::PointXYZ pi;
    if (dis < dis_threshold_min || dis > dis_threshold_max) {
      continue;
    } else {
      if (dis > dis_threshold_min && dis <= dis_threshold_max) {
        pi.x = x;
        pi.y = y;
        pi.z = z;
      }
    }
    Eigen::Vector3d cur_project;

    cur_project[0] = (-A * (B * y + C * z + D) + x * (B * B + C * C)) /
                     (A * A + B * B + C * C);
    cur_project[1] = (-B * (A * x + C * z + D) + y * (A * A + C * C)) /
                     (A * A + B * B + C * C);
    cur_project[2] = (-C * (A * x + B * y + D) + z * (A * A + B * B)) /
                     (A * A + B * B + C * C);
    pcl::PointXYZ p;
    p.x = cur_project[0];
    p.y = cur_project[1];
    p.z = cur_project[2];
    double project_x =
        cur_project[0] * ay + cur_project[1] * by + cur_project[2] * cy + dy;
    double project_y =
        cur_project[0] * ax + cur_project[1] * bx + cur_project[2] * cx + dx;
    Eigen::Vector2d p_2d(project_x, project_y);
    point_list_2d.push_back(p_2d);
    dis_list_2d.push_back(dis);
    point_list_3d.points.push_back(pi);
  }
  double min_x = 10;
  double max_x = -10;
  double min_y = 10;
  double max_y = -10;
  if (point_list_2d.size() <= 5) {
    return;
  }
  for (auto pi : point_list_2d) {
    if (pi[0] < min_x) {
      min_x = pi[0];
    }
    if (pi[0] > max_x) {
      max_x = pi[0];
    }
    if (pi[1] < min_y) {
      min_y = pi[1];
    }
    if (pi[1] > max_y) {
      max_y = pi[1];
    }
  }
  // segment project cloud
  int segmen_base_num = 5;
  double segmen_len = segmen_base_num * resolution;
  int x_segment_num = (max_x - min_x) / segmen_len + 1;
  int y_segment_num = (max_y - min_y) / segmen_len + 1;
  int x_axis_len = (int)((max_x - min_x) / resolution + segmen_base_num);
  int y_axis_len = (int)((max_y - min_y) / resolution + segmen_base_num);

  std::vector<double> **dis_container = new std::vector<double> *[x_axis_len];
  BinaryDescriptor **binary_container = new BinaryDescriptor *[x_axis_len];
  for (int i = 0; i < x_axis_len; i++) {
    dis_container[i] = new std::vector<double>[y_axis_len];
    binary_container[i] = new BinaryDescriptor[y_axis_len];
  }
  double **img_count = new double *[x_axis_len];
  for (int i = 0; i < x_axis_len; i++) {
    img_count[i] = new double[y_axis_len];
  }
  double **dis_array = new double *[x_axis_len];
  for (int i = 0; i < x_axis_len; i++) {
    dis_array[i] = new double[y_axis_len];
  }
  double **mean_x_list = new double *[x_axis_len];
  for (int i = 0; i < x_axis_len; i++) {
    mean_x_list[i] = new double[y_axis_len];
  }
  double **mean_y_list = new double *[x_axis_len];
  for (int i = 0; i < x_axis_len; i++) {
    mean_y_list[i] = new double[y_axis_len];
  }
  for (int x = 0; x < x_axis_len; x++) {
    for (int y = 0; y < y_axis_len; y++) {
      img_count[x][y] = 0;
      mean_x_list[x][y] = 0;
      mean_y_list[x][y] = 0;
      dis_array[x][y] = 0;
      std::vector<double> single_dis_container;
      dis_container[x][y] = single_dis_container;
    }
  }

  for (size_t i = 0; i < point_list_2d.size(); i++) {
    int x_index = (int)((point_list_2d[i][0] - min_x) / resolution);
    int y_index = (int)((point_list_2d[i][1] - min_y) / resolution);
    mean_x_list[x_index][y_index] += point_list_2d[i][0];
    mean_y_list[x_index][y_index] += point_list_2d[i][1];
    img_count[x_index][y_index]++;
    dis_container[x_index][y_index].push_back(dis_list_2d[i]);
  }

  for (int x = 0; x < x_axis_len; x++) {
    for (int y = 0; y < y_axis_len; y++) {
      // calc segment dis array
      if (img_count[x][y] > 0) {
        int cut_num = (dis_threshold_max - dis_threshold_min) / high_inc;
        std::vector<bool> occup_list;
        std::vector<double> cnt_list;
        BinaryDescriptor single_binary;
        for (size_t i = 0; i < cut_num; i++) {
          cnt_list.push_back(0);
          occup_list.push_back(false);
        }
        for (size_t j = 0; j < dis_container[x][y].size(); j++) {
          int cnt_index =
              (dis_container[x][y][j] - dis_threshold_min) / high_inc;
          cnt_list[cnt_index]++;
        }
        double segmnt_dis = 0;
        for (size_t i = 0; i < cut_num; i++) {
          if (cnt_list[i] >= 1) {
            segmnt_dis++;
            occup_list[i] = true;
          }
        }
        dis_array[x][y] = segmnt_dis;
        single_binary.occupy_array_ = occup_list;
        single_binary.summary_ = segmnt_dis;
        binary_container[x][y] = single_binary;
      }
    }
  }
  for (size_t y = 0; y < y_axis_len; y++) {
    for (size_t x = 0; x < x_axis_len; x++) {
      if (img_count[x][y] > 0) {
        // double px = mean_x_list[x][y] / img_count[x][y];
        // double py = mean_y_list[x][y] / img_count[x][y];
        double px = x * resolution + min_x;
        double py = y * resolution + min_y;
        Eigen::Vector3d coord = py * x_axis + px * y_axis + project_center;
        pcl::PointXYZ pi;
        pi.x = coord[0];
        pi.y = coord[1];
        pi.z = coord[2];
        BinaryDescriptor single_binary = binary_container[x][y];
        single_binary.location_ = coord;
        binary_around_list.push_back(single_binary);
      }
    }
  }

  // debug image
  double max_dis_cnt = (dis_threshold_max - dis_threshold_min) / high_inc;
  cv::Mat proj_image = cv::Mat::zeros(y_axis_len, x_axis_len, CV_8UC1);
  for (size_t y = 0; y < y_axis_len; y++) {
    for (size_t x = 0; x < x_axis_len; x++) {
      if (dis_array[x][y] != 0) {
        proj_image.at<uchar>(y, x) = dis_array[x][y] * 20 + 50;
      }
    }
  }
  cv::Mat image_max; // 等比例放大图
  cv::resize(proj_image, image_max,
             cv::Size(y_axis_len * 2, x_axis_len * 2)); // 放大操作
  cv::Mat out;
  // cv::equalizeHist(proj_image, out);
  cv::imshow("proj image", proj_image);
  cv::waitKey();

  // filter by distance
  std::vector<double> max_dis_list;
  std::vector<int> max_dis_x_index_list;
  std::vector<int> max_dis_y_index_list;

  for (int x_segment_index = 0; x_segment_index < x_segment_num;
       x_segment_index++) {
    for (int y_segment_index = 0; y_segment_index < y_segment_num;
         y_segment_index++) {
      double max_dis = 0;
      int max_dis_x_index = -10;
      int max_dis_y_index = -10;
      for (int x_index = x_segment_index * segmen_base_num;
           x_index < (x_segment_index + 1) * segmen_base_num; x_index++) {
        for (int y_index = y_segment_index * segmen_base_num;
             y_index < (y_segment_index + 1) * segmen_base_num; y_index++) {
          if (dis_array[x_index][y_index] > max_dis) {
            max_dis = dis_array[x_index][y_index];
            max_dis_x_index = x_index;
            max_dis_y_index = y_index;
          }
        }
      }
      if (max_dis >= binary_min_dis) {
        bool is_touch = true;
        // is_touch =
        //     binary_container[max_dis_x_index][max_dis_y_index]
        //         .occupy_array_[0] ||
        //     binary_container[max_dis_x_index][max_dis_y_index]
        //         .occupy_array_[1] ||
        //     binary_container[max_dis_x_index][max_dis_y_index]
        //         .occupy_array_[2] ||
        //     binary_container[max_dis_x_index][max_dis_y_index].occupy_array_[3];
        if (is_touch) {
          max_dis_list.push_back(max_dis);
          max_dis_x_index_list.push_back(max_dis_x_index);
          max_dis_y_index_list.push_back(max_dis_y_index);
        }
      }
    }
  }
  // calc line or not
  std::vector<Eigen::Vector2i> direction_list;
  Eigen::Vector2i d(0, 1);
  direction_list.push_back(d);
  d << 1, 0;
  direction_list.push_back(d);
  d << 1, 1;
  direction_list.push_back(d);
  d << 1, -1;
  direction_list.push_back(d);
  for (size_t i = 0; i < max_dis_list.size(); i++) {
    Eigen::Vector2i p(max_dis_x_index_list[i], max_dis_y_index_list[i]);
    if (p[0] <= 0 || p[0] >= x_axis_len - 1 || p[1] <= 0 ||
        p[1] >= y_axis_len - 1) {
      continue;
    }
    bool is_add = true;
    // if (line_filter_enable) {
    //   for (int dx = -1; dx <= 1; dx++) {
    //     for (int dy = -1; dy <= 1; dy++) {
    //       if (dx == 0 && dy == 0) {
    //         continue;
    //       }
    //       Eigen::Vector2i p_near = p;
    //       p_near[0] = p[0] + dx;
    //       p_near[1] = p[1] + dy;
    //       double threshold = dis_array[p[0]][p[1]] - 3;
    //       if (dis_array[p_near[0]][p_near[1]] >= threshold) {
    //         is_add = false;
    //       }
    //     }
    //   }
    // }

    if (line_filter_enable) {
      for (int j = 0; j < 4; j++) {
        Eigen::Vector2i p(max_dis_x_index_list[i], max_dis_y_index_list[i]);
        if (p[0] <= 0 || p[0] >= x_axis_len - 1 || p[1] <= 0 ||
            p[1] >= y_axis_len - 1) {
          continue;
        }
        Eigen::Vector2i p1 = p + direction_list[j];
        Eigen::Vector2i p2 = p - direction_list[j];
        double threshold = dis_array[p[0]][p[1]] - 3;
        if (dis_array[p1[0]][p1[1]] >= threshold) {
          if (dis_array[p2[0]][p2[1]] >= 0.5 * dis_array[p[0]][p[1]]) {
            is_add = false;
          }
        }
        if (dis_array[p2[0]][p2[1]] >= threshold) {
          if (dis_array[p1[0]][p1[1]] >= 0.5 * dis_array[p[0]][p[1]]) {
            is_add = false;
          }
        }
        if (dis_array[p1[0]][p1[1]] >= threshold) {
          if (dis_array[p2[0]][p2[1]] >= threshold) {
            is_add = false;
          }
        }
        if (dis_array[p2[0]][p2[1]] >= threshold) {
          if (dis_array[p1[0]][p1[1]] >= threshold) {
            is_add = false;
          }
        }
      }
    }

    if (is_add) {
      double px =
          mean_x_list[max_dis_x_index_list[i]][max_dis_y_index_list[i]] /
          img_count[max_dis_x_index_list[i]][max_dis_y_index_list[i]];
      double py =
          mean_y_list[max_dis_x_index_list[i]][max_dis_y_index_list[i]] /
          img_count[max_dis_x_index_list[i]][max_dis_y_index_list[i]];
      Eigen::Vector3d coord = py * x_axis + px * y_axis + project_center;
      pcl::PointXYZ pi;
      pi.x = coord[0];
      pi.y = coord[1];
      pi.z = coord[2];
      BinaryDescriptor single_binary =
          binary_container[max_dis_x_index_list[i]][max_dis_y_index_list[i]];
      single_binary.location_ = coord;
      binary_list.push_back(single_binary);
    }
  }
  for (int i = 0; i < x_axis_len; i++) {
    delete[] binary_container[i];
    delete[] dis_container[i];
    delete[] img_count[i];
    delete[] dis_array[i];
    delete[] mean_x_list[i];
    delete[] mean_y_list[i];
  }
  delete[] binary_container;
  delete[] dis_container;
  delete[] img_count;
  delete[] dis_array;
  delete[] mean_x_list;
  delete[] mean_y_list;
}

void extract_binary_all(const ConfigSetting &config_setting,
                        const Eigen::Vector3d &project_center,
                        const Eigen::Vector3d &project_normal,
                        const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
                        std::vector<BinaryDescriptor> &binary_list,
                        cv::Mat &binary_image) {
  binary_list.clear();
  double binary_min_dis = config_setting.summary_min_thre_;
  double resolution = config_setting.proj_image_resolution_;
  double dis_threshold_min = config_setting.proj_dis_min_;
  double dis_threshold_max = config_setting.proj_dis_max_;
  double high_inc = config_setting.proj_image_high_inc_;
  bool line_filter_enable = config_setting.line_filter_enable_;
  double A = project_normal[0];
  double B = project_normal[1];
  double C = project_normal[2];
  double D =
      -(A * project_center[0] + B * project_center[1] + C * project_center[2]);
  std::vector<Eigen::Vector3d> projection_points;
  Eigen::Vector3d x_axis(1, 100, 0);
  if (C != 0) {
    x_axis[2] = -(A + B) / C;
  } else if (B != 0) {
    x_axis[1] = -A / B;
  } else {
    x_axis[0] = 0;
    x_axis[1] = 1;
  }
  x_axis.normalize();
  Eigen::Vector3d y_axis = project_normal.cross(x_axis);
  y_axis.normalize();
  double ax = x_axis[0];
  double bx = x_axis[1];
  double cx = x_axis[2];
  double dx = -(ax * project_center[0] + bx * project_center[1] +
                cx * project_center[2]);
  double ay = y_axis[0];
  double by = y_axis[1];
  double cy = y_axis[2];
  double dy = -(ay * project_center[0] + by * project_center[1] +
                cy * project_center[2]);
  std::vector<Eigen::Vector2d> point_list_2d;
  pcl::PointCloud<pcl::PointXYZ> point_list_3d;
  std::vector<double> dis_list_2d;
  for (size_t i = 0; i < input_cloud->size(); i++) {
    double x = input_cloud->points[i].x;
    double y = input_cloud->points[i].y;
    double z = input_cloud->points[i].z;
    double dis = fabs(x * A + y * B + z * C + D);
    pcl::PointXYZ pi;
    if (dis < dis_threshold_min || dis > dis_threshold_max) {
      continue;
    } else {
      if (dis > dis_threshold_min && dis <= dis_threshold_max) {
        pi.x = x;
        pi.y = y;
        pi.z = z;
      }
    }
    Eigen::Vector3d cur_project;

    cur_project[0] = (-A * (B * y + C * z + D) + x * (B * B + C * C)) /
                     (A * A + B * B + C * C);
    cur_project[1] = (-B * (A * x + C * z + D) + y * (A * A + C * C)) /
                     (A * A + B * B + C * C);
    cur_project[2] = (-C * (A * x + B * y + D) + z * (A * A + B * B)) /
                     (A * A + B * B + C * C);
    pcl::PointXYZ p;
    p.x = cur_project[0];
    p.y = cur_project[1];
    p.z = cur_project[2];
    double project_x =
        cur_project[0] * ay + cur_project[1] * by + cur_project[2] * cy + dy;
    double project_y =
        cur_project[0] * ax + cur_project[1] * bx + cur_project[2] * cx + dx;
    Eigen::Vector2d p_2d(project_x, project_y);
    point_list_2d.push_back(p_2d);
    dis_list_2d.push_back(dis);
    point_list_3d.points.push_back(pi);
  }
  double min_x = 10;
  double max_x = -10;
  double min_y = 10;
  double max_y = -10;
  if (point_list_2d.size() <= 5) {
    return;
  }
  for (auto pi : point_list_2d) {
    if (pi[0] < min_x) {
      min_x = pi[0];
    }
    if (pi[0] > max_x) {
      max_x = pi[0];
    }
    if (pi[1] < min_y) {
      min_y = pi[1];
    }
    if (pi[1] > max_y) {
      max_y = pi[1];
    }
  }
  // segment project cloud
  int segmen_base_num = 5;
  double segmen_len = segmen_base_num * resolution;
  int x_segment_num = (max_x - min_x) / segmen_len + 1;
  int y_segment_num = (max_y - min_y) / segmen_len + 1;
  int x_axis_len = (int)((max_x - min_x) / resolution + segmen_base_num);
  int y_axis_len = (int)((max_y - min_y) / resolution + segmen_base_num);

  std::vector<double> **dis_container = new std::vector<double> *[x_axis_len];
  BinaryDescriptor **binary_container = new BinaryDescriptor *[x_axis_len];
  for (int i = 0; i < x_axis_len; i++) {
    dis_container[i] = new std::vector<double>[y_axis_len];
    binary_container[i] = new BinaryDescriptor[y_axis_len];
  }
  double **img_count = new double *[x_axis_len];
  for (int i = 0; i < x_axis_len; i++) {
    img_count[i] = new double[y_axis_len];
  }
  double **dis_array = new double *[x_axis_len];
  for (int i = 0; i < x_axis_len; i++) {
    dis_array[i] = new double[y_axis_len];
  }
  double **mean_x_list = new double *[x_axis_len];
  for (int i = 0; i < x_axis_len; i++) {
    mean_x_list[i] = new double[y_axis_len];
  }
  double **mean_y_list = new double *[x_axis_len];
  for (int i = 0; i < x_axis_len; i++) {
    mean_y_list[i] = new double[y_axis_len];
  }
  for (int x = 0; x < x_axis_len; x++) {
    for (int y = 0; y < y_axis_len; y++) {
      img_count[x][y] = 0;
      mean_x_list[x][y] = 0;
      mean_y_list[x][y] = 0;
      dis_array[x][y] = 0;
      std::vector<double> single_dis_container;
      dis_container[x][y] = single_dis_container;
    }
  }

  for (size_t i = 0; i < point_list_2d.size(); i++) {
    int x_index = (int)((point_list_2d[i][0] - min_x) / resolution);
    int y_index = (int)((point_list_2d[i][1] - min_y) / resolution);
    mean_x_list[x_index][y_index] += point_list_2d[i][0];
    mean_y_list[x_index][y_index] += point_list_2d[i][1];
    img_count[x_index][y_index]++;
    dis_container[x_index][y_index].push_back(dis_list_2d[i]);
  }

  for (int x = 0; x < x_axis_len; x++) {
    for (int y = 0; y < y_axis_len; y++) {
      // calc segment dis array
      if (img_count[x][y] > 0) {
        int cut_num = (dis_threshold_max - dis_threshold_min) / high_inc;
        std::vector<bool> occup_list;
        std::vector<double> cnt_list;
        BinaryDescriptor single_binary;
        for (size_t i = 0; i < cut_num; i++) {
          cnt_list.push_back(0);
          occup_list.push_back(false);
        }
        for (size_t j = 0; j < dis_container[x][y].size(); j++) {
          int cnt_index =
              (dis_container[x][y][j] - dis_threshold_min) / high_inc;
          cnt_list[cnt_index]++;
        }
        double segmnt_dis = 0;
        for (size_t i = 0; i < cut_num; i++) {
          if (cnt_list[i] >= 1) {
            segmnt_dis++;
            occup_list[i] = true;
          }
        }
        dis_array[x][y] = segmnt_dis;
        single_binary.occupy_array_ = occup_list;
        single_binary.summary_ = segmnt_dis;
        binary_container[x][y] = single_binary;
      }
    }
  }
  for (size_t y = 0; y < y_axis_len; y++) {
    for (size_t x = 0; x < x_axis_len; x++) {
      if (img_count[x][y] > 0) {
        // double px = mean_x_list[x][y] / img_count[x][y];
        // double py = mean_y_list[x][y] / img_count[x][y];
        double px = x * resolution + min_x;
        double py = y * resolution + min_y;
        Eigen::Vector3d coord = py * x_axis + px * y_axis + project_center;
        pcl::PointXYZ pi;
        pi.x = coord[0];
        pi.y = coord[1];
        pi.z = coord[2];
        BinaryDescriptor single_binary = binary_container[x][y];
        single_binary.location_ = coord;
        binary_list.push_back(single_binary);
      }
    }
  }

  // debug image
  double max_dis_cnt = (dis_threshold_max - dis_threshold_min) / high_inc;
  cv::Mat proj_image = cv::Mat::zeros(y_axis_len, x_axis_len, CV_8UC1);
  for (size_t y = 0; y < y_axis_len; y++) {
    for (size_t x = 0; x < x_axis_len; x++) {
      if (dis_array[x][y] != 0) {
        proj_image.at<uchar>(y, x) = dis_array[x][y] * 20 + 50;
      }
    }
  }
  cv::Mat image_max; // 等比例放大图
  cv::resize(proj_image, image_max,
             cv::Size(y_axis_len * 2, x_axis_len * 2)); // 放大操作
  binary_image = proj_image;
  // cv::equalizeHist(proj_image, out);
  // cv::imshow("proj image", proj_image);
  // cv::waitKey();
}

void generate_std(const ConfigSetting &config_setting,
                  const std::vector<BinaryDescriptor> &binary_list,
                  const int &frame_number, std::vector<STD> &std_list) {
  double scale = 1.0 / config_setting.std_side_resolution_;
  std::unordered_map<VOXEL_LOC, bool> feat_map;
  pcl::PointCloud<pcl::PointXYZ> key_cloud;
  for (auto var : binary_list) {
    pcl::PointXYZ pi;
    pi.x = var.location_[0];
    pi.y = var.location_[1];
    pi.z = var.location_[2];
    key_cloud.push_back(pi);
  }
  pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kd_tree(
      new pcl::KdTreeFLANN<pcl::PointXYZ>);
  kd_tree->setInputCloud(key_cloud.makeShared());
  int K = config_setting.descriptor_near_num_;
  std::vector<int> pointIdxNKNSearch(K);
  std::vector<float> pointNKNSquaredDistance(K);
  for (size_t i = 0; i < key_cloud.size(); i++) {
    pcl::PointXYZ searchPoint = key_cloud.points[i];
    if (kd_tree->nearestKSearch(searchPoint, K, pointIdxNKNSearch,
                                pointNKNSquaredDistance) > 0) {
      for (int m = 1; m < K - 1; m++) {
        for (int n = m + 1; n < K; n++) {
          pcl::PointXYZ p1 = searchPoint;
          pcl::PointXYZ p2 = key_cloud.points[pointIdxNKNSearch[m]];
          pcl::PointXYZ p3 = key_cloud.points[pointIdxNKNSearch[n]];
          double a = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) +
                          pow(p1.z - p2.z, 2));
          double b = sqrt(pow(p1.x - p3.x, 2) + pow(p1.y - p3.y, 2) +
                          pow(p1.z - p3.z, 2));
          double c = sqrt(pow(p3.x - p2.x, 2) + pow(p3.y - p2.y, 2) +
                          pow(p3.z - p2.z, 2));
          if (a > config_setting.descriptor_max_len_ ||
              b > config_setting.descriptor_max_len_ ||
              c > config_setting.descriptor_max_len_ ||
              a < config_setting.descriptor_min_len_ ||
              b < config_setting.descriptor_min_len_ ||
              c < config_setting.descriptor_min_len_) {
            continue;
          }
          double temp;
          Eigen::Vector3d A, B, C;
          Eigen::Vector3i l1, l2, l3;
          Eigen::Vector3i l_temp;
          l1 << 1, 2, 0;
          l2 << 1, 0, 3;
          l3 << 0, 2, 3;
          if (a > b) {
            temp = a;
            a = b;
            b = temp;
            l_temp = l1;
            l1 = l2;
            l2 = l_temp;
          }
          if (b > c) {
            temp = b;
            b = c;
            c = temp;
            l_temp = l2;
            l2 = l3;
            l3 = l_temp;
          }
          if (a > b) {
            temp = a;
            a = b;
            b = temp;
            l_temp = l1;
            l1 = l2;
            l2 = l_temp;
          }
          if (fabs(c - (a + b)) < 0.2) {
            continue;
          }

          pcl::PointXYZ d_p;
          d_p.x = a * 1000;
          d_p.y = b * 1000;
          d_p.z = c * 1000;
          VOXEL_LOC position((int64_t)d_p.x, (int64_t)d_p.y, (int64_t)d_p.z);
          auto iter = feat_map.find(position);
          Eigen::Vector3d normal_1, normal_2, normal_3;
          BinaryDescriptor binary_A;
          BinaryDescriptor binary_B;
          BinaryDescriptor binary_C;
          if (iter == feat_map.end()) {
            if (l1[0] == l2[0]) {
              A << p1.x, p1.y, p1.z;
              binary_A = binary_list[i];
            } else if (l1[1] == l2[1]) {
              A << p2.x, p2.y, p2.z;
              binary_A = binary_list[pointIdxNKNSearch[m]];
            } else {
              A << p3.x, p3.y, p3.z;
              binary_A = binary_list[pointIdxNKNSearch[n]];
            }
            if (l1[0] == l3[0]) {
              B << p1.x, p1.y, p1.z;
              binary_B = binary_list[i];
            } else if (l1[1] == l3[1]) {
              B << p2.x, p2.y, p2.z;
              binary_B = binary_list[pointIdxNKNSearch[m]];
            } else {
              B << p3.x, p3.y, p3.z;
              binary_B = binary_list[pointIdxNKNSearch[n]];
            }
            if (l2[0] == l3[0]) {
              C << p1.x, p1.y, p1.z;
              binary_C = binary_list[i];
            } else if (l2[1] == l3[1]) {
              C << p2.x, p2.y, p2.z;
              binary_C = binary_list[pointIdxNKNSearch[m]];
            } else {
              C << p3.x, p3.y, p3.z;
              binary_C = binary_list[pointIdxNKNSearch[n]];
            }
            STD single_descriptor;
            single_descriptor.binary_A_ = binary_A;
            single_descriptor.binary_B_ = binary_B;
            single_descriptor.binary_C_ = binary_C;
            single_descriptor.center_ = (A + B + C) / 3;
            single_descriptor.triangle_ << scale * a, scale * b, scale * c;
            single_descriptor.angle_[0] = fabs(5 * normal_1.dot(normal_2));
            single_descriptor.angle_[1] = fabs(5 * normal_1.dot(normal_3));
            single_descriptor.angle_[2] = fabs(5 * normal_3.dot(normal_2));
            // single_descriptor.angle << 0, 0, 0;
            single_descriptor.frame_number_ = frame_number;
            // single_descriptor.score_frame_.push_back(frame_number);
            Eigen::Matrix3d triangle_positon;
            triangle_positon.block<3, 1>(0, 0) = A;
            triangle_positon.block<3, 1>(0, 1) = B;
            triangle_positon.block<3, 1>(0, 2) = C;
            // single_descriptor.position_list_.push_back(triangle_positon);
            // single_descriptor.triangle_scale_ = scale;
            feat_map[position] = true;
            std_list.push_back(single_descriptor);
          }
        }
      }
    }
  }
}

void candidate_searcher(
    const ConfigSetting &config_setting,
    std::unordered_map<STD_LOC, std::vector<STD>> &descriptor_map,
    std::vector<STD> &current_STD_list,
    std::vector<STDMatchList> &alternative_match) {

  int outlier = 0;
  double max_dis = 200;
  double match_array[20000] = {0};
  std::vector<std::pair<STD, STD>> match_list;
  std::vector<int> match_list_index;
  std::vector<Eigen::Vector3i> voxel_round;
  Eigen::Vector3i voxel_inc(0, 0, 0);
  voxel_round.push_back(voxel_inc);
  // for (int x = -1; x <= 1; x++) {
  //   for (int y = -1; y <= 1; y++) {
  //     for (int z = -1; z <= 1; z++) {
  //       Eigen::Vector3i voxel_inc(x, y, z);
  //       voxel_round.push_back(voxel_inc);
  //     }
  //   }
  // }
  std::vector<bool> useful_match(current_STD_list.size());
  std::vector<std::vector<size_t>> useful_match_index(current_STD_list.size());
  std::vector<std::vector<STD_LOC>> useful_match_position(
      current_STD_list.size());
  std::vector<STD_LOC> useful_match_single_position(current_STD_list.size());
  std::vector<size_t> index(current_STD_list.size());
  for (size_t i = 0; i < index.size(); ++i) {
    index[i] = i;
    useful_match[i] = false;
  }
  std::mutex mylock;
  auto t0 = std::chrono::high_resolution_clock::now();

  int query_num = 0;
  int pass_num = 0;
  std::for_each(std::execution::par_unseq, index.begin(), index.end(),
                [&](const size_t &i) {
                  STD descriptor = current_STD_list[i];
                  STD_LOC position;
                  int best_index = 0;
                  STD_LOC best_position;

                  for (auto voxel_inc : voxel_round) {
                    position.x = (int)(descriptor.triangle_[0] + 0.5);
                    position.y = (int)(descriptor.triangle_[1] + 0.5);
                    position.z = (int)(descriptor.triangle_[2] + 0.5);
                    Eigen::Vector3d voxel_center((double)position.x + 0.5,
                                                 (double)position.y + 0.5,
                                                 (double)position.z + 0.5);
                    auto iter = descriptor_map.find(position);
                    if (iter != descriptor_map.end()) {
                      mylock.lock();
                      // pass_num++;
                      useful_match[i] = true;
                      useful_match_single_position[i] = position;
                      mylock.unlock();
                      // for (size_t j = 0; j <
                      // descriptor_map[position].size(); j++) {
                      //   // mylock.lock();
                      //   // query_num++;
                      //   // mylock.unlock();
                      //   // if ((descriptor.center_ -
                      //   // descriptor_map[position][j].center_)
                      //   //         .norm() < max_dis)
                      //   if ((descriptor.frame_number_ -
                      //        descriptor_map[position][j].frame_number_) >
                      //       config_setting.skip_near_num_) {
                      //     // double dis = (descriptor.triangle_ -
                      //     // descriptor_map[position][j].triangle_)
                      //     //                  .norm();
                      //     double dis = 0;
                      //     if (dis < dis_threshold) {
                      //       mylock.lock();
                      //       // pass_num++;
                      //       useful_match[i] = true;
                      //       useful_match_position[i].push_back(position);
                      //       useful_match_index[i].push_back(j);
                      //       mylock.unlock();
                      //     }
                      //   }
                      // }
                    }
                  }
                });
  // std::cout << "query num: " << query_num << " , pass num: " << pass_num
  //           << std::endl;

  int useful_match_size = 0;
  for (size_t i = 0; i < useful_match.size(); i++) {
    if (useful_match[i]) {
      useful_match_size++;
    }
  }
  std::cout << "descriptor size:" << current_STD_list.size()
            << ", useful_match_size: " << useful_match_size << std::endl;
  auto t1 = std::chrono::high_resolution_clock::now();
  std::for_each(
      std::execution::par_unseq, index.begin(), index.end(),
      [&](const size_t &i) {
        if (useful_match[i]) {
          std::pair<STD, STD> single_match_pair;
          single_match_pair.first = current_STD_list[i];
          for (size_t j = 0;
               j < descriptor_map[useful_match_single_position[i]].size();
               j++) {
            if ((current_STD_list[i].frame_number_ -
                 descriptor_map[useful_match_single_position[i]][j]
                     .frame_number_) <= config_setting.skip_near_num_) {
              continue;
            }
            double dis_threshold =
                current_STD_list[i].triangle_.norm() *
                config_setting.rough_dis_threshold_; // old 0.005
            double dis =
                (current_STD_list[i].triangle_ -
                 descriptor_map[useful_match_single_position[i]][j].triangle_)
                    .norm();
            if (dis > dis_threshold) {
              continue;
            }
            // if ((current_STD_list[i].center_ -
            //      descriptor_map[useful_match_single_position[i]][j].center_)
            //         .norm() > max_dis) {
            //   continue;
            // }
            double similarity =
                (binary_similarity(
                     current_STD_list[i].binary_A_,
                     descriptor_map[useful_match_single_position[i]][j]
                         .binary_A_) +
                 binary_similarity(
                     current_STD_list[i].binary_B_,
                     descriptor_map[useful_match_single_position[i]][j]
                         .binary_B_) +
                 binary_similarity(
                     current_STD_list[i].binary_C_,
                     descriptor_map[useful_match_single_position[i]][j]
                         .binary_C_)) /
                3;
            if (similarity > config_setting.similarity_threshold_) {
              single_match_pair.second =
                  descriptor_map[useful_match_single_position[i]][j];
              mylock.lock();
              match_array[single_match_pair.second.frame_number_] += similarity;
              match_list.push_back(single_match_pair);
              match_list_index.push_back(
                  single_match_pair.second.frame_number_);
              mylock.unlock();
            }
          }

          // for (size_t j = 0; j < useful_match_index[i].size(); j++)
          // {
          //   single_match_pair.second =
          //       descriptor_map[useful_match_position[i][j]]
          //                     [useful_match_index[i][j]];
          //   double similarity =
          //       (binary_similarity(single_match_pair.first.binary_A_,
          //                          single_match_pair.second.binary_A_)
          //                          +
          //        binary_similarity(single_match_pair.first.binary_B_,
          //                          single_match_pair.second.binary_B_)
          //                          +
          //        binary_similarity(single_match_pair.first.binary_C_,
          //                          single_match_pair.second.binary_C_))
          //                          /
          //       3;
          //   if (similarity > 0.7) {
          //     mylock.lock();
          //     match_array[single_match_pair.second.frame_number_]
          //     += similarity;
          //     match_list.push_back(single_match_pair);
          //     match_list_index.push_back(
          //         single_match_pair.second.frame_number_);
          //     mylock.unlock();
          //   }
          // }
        }
      });
  auto t2 = std::chrono::high_resolution_clock::now();
  for (int cnt = 0; cnt < config_setting.candidate_num_; cnt++) {
    double max_vote = 1;
    int max_vote_index = -1;
    for (int i = 0; i < 20000; i++) {
      if (match_array[i] > max_vote) {
        max_vote = match_array[i];
        max_vote_index = i;
      }
    }
    STDMatchList match_triangle_list;
    if (max_vote_index >= 0) {
      match_array[max_vote_index] = 0;
      match_triangle_list.match_frame_ = max_vote_index;
      double mean_dis = 0;
      for (size_t i = 0; i < match_list_index.size(); i++) {
        if (match_list_index[i] == max_vote_index) {
          match_triangle_list.match_list_.push_back(match_list[i]);
        }
      }
    }
    alternative_match.push_back(match_triangle_list);
  }
  auto t3 = std::chrono::high_resolution_clock::now();
  std::cout << " rough part1 time:"
            << std::chrono::duration_cast<std::chrono::duration<double>>(t1 -
                                                                         t0)
                       .count() *
                   1000
            << " ms" << std::endl;
  std::cout << " rough part2 time:"
            << std::chrono::duration_cast<std::chrono::duration<double>>(t2 -
                                                                         t1)
                       .count() *
                   1000
            << " ms" << std::endl;
  std::cout << " rough part3 time:"
            << std::chrono::duration_cast<std::chrono::duration<double>>(t3 -
                                                                         t2)
                       .count() *
                   1000
            << " ms" << std::endl;
}

void candidate_searcher_old(
    const ConfigSetting &config_setting,
    std::unordered_map<STD_LOC, std::vector<STD>> &descriptor_map,
    std::vector<STD> &current_STD_list,
    std::vector<STDMatchList> &alternative_match) {

  int outlier = 0;
  double max_dis = 100;
  double match_array[20000] = {0};
  std::vector<std::pair<STD, STD>> match_list;
  std::vector<int> match_list_index;
  std::vector<Eigen::Vector3i> voxel_round;
  for (int x = -1; x <= 1; x++) {
    for (int y = -1; y <= 1; y++) {
      for (int z = -1; z <= 1; z++) {
        Eigen::Vector3i voxel_inc(x, y, z);
        voxel_round.push_back(voxel_inc);
      }
    }
  }
  std::vector<bool> useful_match(current_STD_list.size());
  std::vector<std::vector<size_t>> useful_match_index(current_STD_list.size());
  std::vector<std::vector<STD_LOC>> useful_match_position(
      current_STD_list.size());
  std::vector<size_t> index(current_STD_list.size());
  for (size_t i = 0; i < index.size(); ++i) {
    index[i] = i;
    useful_match[i] = false;
  }
  std::mutex mylock;
  auto t0 = std::chrono::high_resolution_clock::now();

  int query_num = 0;
  int pass_num = 0;
  std::for_each(
      std::execution::par_unseq, index.begin(), index.end(),
      [&](const size_t &i) {
        STD descriptor = current_STD_list[i];
        STD_LOC position;
        int best_index = 0;
        STD_LOC best_position;
        double dis_threshold = descriptor.triangle_.norm() *
                               config_setting.rough_dis_threshold_; // old 0.005
        for (auto voxel_inc : voxel_round) {
          position.x = (int)(descriptor.triangle_[0] + voxel_inc[0]);
          position.y = (int)(descriptor.triangle_[1] + voxel_inc[1]);
          position.z = (int)(descriptor.triangle_[2] + voxel_inc[2]);
          Eigen::Vector3d voxel_center((double)position.x + 0.5,
                                       (double)position.y + 0.5,
                                       (double)position.z + 0.5);
          if ((descriptor.triangle_ - voxel_center).norm() < 1.5) {
            auto iter = descriptor_map.find(position);
            if (iter != descriptor_map.end()) {
              bool is_push_position = false;
              for (size_t j = 0; j < descriptor_map[position].size(); j++) {
                if ((descriptor.frame_number_ -
                     descriptor_map[position][j].frame_number_) >
                    config_setting.skip_near_num_) {
                  // if ((descriptor.center_ -
                  // descriptor_map[position][j].center_)
                  //         .norm() > max_dis) {
                  //   continue;
                  // }
                  double dis = (descriptor.triangle_ -
                                descriptor_map[position][j].triangle_)
                                   .norm();
                  if (dis < dis_threshold) {
                    double similarity =
                        (binary_similarity(
                             descriptor.binary_A_,
                             descriptor_map[position][j].binary_A_) +
                         binary_similarity(
                             descriptor.binary_B_,
                             descriptor_map[position][j].binary_B_) +
                         binary_similarity(
                             descriptor.binary_C_,
                             descriptor_map[position][j].binary_C_)) /
                        3;
                    if (similarity > config_setting.similarity_threshold_) {
                      // if (!is_push_position) {
                      //   useful_match[i] = true;
                      //   useful_match_position[i].push_back(position);
                      //   is_push_position = true;
                      // }
                      useful_match[i] = true;
                      useful_match_position[i].push_back(position);
                      useful_match_index[i].push_back(j);
                    }
                    // mylock.lock();
                    // pass_num++;
                    // mylock.unlock();
                  }
                }
              }
            }
          }
        }
      });
  std::vector<Eigen::Vector2i, Eigen::aligned_allocator<Eigen::Vector2i>>
      index_recorder;
  auto t1 = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < useful_match.size(); i++) {
    if (useful_match[i]) {

      // std::pair<STD, STD> single_match_pair;
      // single_match_pair.first = current_STD_list[i];
      for (size_t j = 0; j < useful_match_index[i].size(); j++) {
        // single_match_pair.second =
        // descriptor_map[useful_match_position[i][j]]
        //                                          [useful_match_index[i][j]];
        match_array[descriptor_map[useful_match_position[i][j]]
                                  [useful_match_index[i][j]]
                                      .frame_number_] += 1;
        Eigen::Vector2i match_index(i, j);
        index_recorder.push_back(match_index);
        // match_list.push_back(single_match_pair);
        match_list_index.push_back(descriptor_map[useful_match_position[i][j]]
                                                 [useful_match_index[i][j]]
                                                     .frame_number_);
      }
    }
  }
  bool multi_thread_en = false;
  if (multi_thread_en) {
    std::for_each(
        std::execution::par_unseq, index.begin(), index.end(),
        [&](const size_t &i) {
          if (useful_match[i]) {
            std::pair<STD, STD> single_match_pair;
            single_match_pair.first = current_STD_list[i];
            for (size_t j = 0; j < useful_match_index[i].size(); j++) {
              single_match_pair.second =
                  descriptor_map[useful_match_position[i][j]]
                                [useful_match_index[i][j]];
              // double similarity =
              //     (binary_similarity(single_match_pair.first.binary_A_,
              //                        single_match_pair.second.binary_A_)
              //                        +
              //      binary_similarity(single_match_pair.first.binary_B_,
              //                        single_match_pair.second.binary_B_)
              //                        +
              //      binary_similarity(single_match_pair.first.binary_C_,
              //                        single_match_pair.second.binary_C_))
              //                        /
              //     3;
              // if (similarity > 0.7) {
              //   mylock.lock();
              //   match_array[single_match_pair.second.frame_number_]
              //   += similarity;
              //   match_list.push_back(single_match_pair);
              //   match_list_index.push_back(
              //       single_match_pair.second.frame_number_);
              //   mylock.unlock();
              // }
              mylock.lock();
              match_array[single_match_pair.second.frame_number_] += 1;
              match_list.push_back(single_match_pair);
              match_list_index.push_back(
                  single_match_pair.second.frame_number_);
              mylock.unlock();
            }
          }
        });
  }

  auto t2 = std::chrono::high_resolution_clock::now();
  // std::cout << "prepare match list size: " << match_list_index.size()
  //           << std::endl;
  // use index recorder

  for (int cnt = 0; cnt < config_setting.candidate_num_; cnt++) {
    double max_vote = 1;
    int max_vote_index = -1;
    for (int i = 0; i < 20000; i++) {
      if (match_array[i] > max_vote) {
        max_vote = match_array[i];
        max_vote_index = i;
      }
    }
    STDMatchList match_triangle_list;
    if (max_vote_index >= 0) {
      match_array[max_vote_index] = 0;
      match_triangle_list.match_frame_ = max_vote_index;
      double mean_dis = 0;
      for (size_t i = 0; i < index_recorder.size(); i++) {
        if (match_list_index[i] == max_vote_index) {
          std::pair<STD, STD> single_match_pair;
          single_match_pair.first = current_STD_list[index_recorder[i][0]];
          single_match_pair.second =
              descriptor_map[useful_match_position[index_recorder[i][0]]
                                                  [index_recorder[i][1]]]
                            [useful_match_index[index_recorder[i][0]]
                                               [index_recorder[i][1]]];
          match_triangle_list.match_list_.push_back(single_match_pair);
        }
      }
    }
    alternative_match.push_back(match_triangle_list);
  }
  // for (int cnt = 0; cnt < config_setting.candidate_num_; cnt++) {
  //   double max_vote = 1;
  //   int max_vote_index = -1;
  //   for (int i = 0; i < 10000; i++) {
  //     if (match_array[i] > max_vote) {
  //       max_vote = match_array[i];
  //       max_vote_index = i;
  //     }
  //   }
  //   STDMatchList match_triangle_list;
  //   if (max_vote_index >= 0) {
  //     match_array[max_vote_index] = 0;
  //     match_triangle_list.match_frame_ = max_vote_index;
  //     double mean_dis = 0;
  //     for (size_t i = 0; i < match_list_index.size(); i++) {
  //       if (match_list_index[i] == max_vote_index) {
  //         match_triangle_list.match_list_.push_back(match_list[i]);
  //       }
  //     }
  //   }
  //   alternative_match.push_back(match_triangle_list);
  // }
  auto t3 = std::chrono::high_resolution_clock::now();
  // std::cout << " rough part1 time:"
  //           << std::chrono::duration_cast<std::chrono::duration<double>>(t1 -
  //                                                                        t0)
  //                      .count() *
  //                  1000
  //           << " ms" << std::endl;
  // std::cout << " rough part2 time:"
  //           << std::chrono::duration_cast<std::chrono::duration<double>>(t2 -
  //                                                                        t1)
  //                      .count() *
  //                  1000
  //           << " ms" << std::endl;
  // std::cout << " rough part3 time:"
  //           << std::chrono::duration_cast<std::chrono::duration<double>>(t3 -
  //                                                                        t2)
  //                      .count() *
  //                  1000
  //           << " ms" << std::endl;
  // getchar();
}

void triangle_solver(std::pair<STD, STD> &std_pair, Eigen::Matrix3d &std_rot,
                     Eigen::Vector3d &std_t) {
  Eigen::Matrix3d src = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d ref = Eigen::Matrix3d::Zero();
  src.col(0) = std_pair.first.binary_A_.location_ - std_pair.first.center_;
  src.col(1) = std_pair.first.binary_B_.location_ - std_pair.first.center_;
  src.col(2) = std_pair.first.binary_C_.location_ - std_pair.first.center_;
  ref.col(0) = std_pair.second.binary_A_.location_ - std_pair.second.center_;
  ref.col(1) = std_pair.second.binary_B_.location_ - std_pair.second.center_;
  ref.col(2) = std_pair.second.binary_C_.location_ - std_pair.second.center_;
  Eigen::Matrix3d covariance = src * ref.transpose();
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(covariance, Eigen::ComputeThinU |
                                                        Eigen::ComputeThinV);
  Eigen::Matrix3d V = svd.matrixV();
  Eigen::Matrix3d U = svd.matrixU();
  std_rot = V * U.transpose();
  if (std_rot.determinant() < 0) {
    Eigen::Matrix3d K;
    K << 1, 0, 0, 0, 1, 0, 0, 0, -1;
    std_rot = V * K * U.transpose();
  }
  std_t = -std_rot * std_pair.first.center_ + std_pair.second.center_;
}

void fine_loop_detection(
    const ConfigSetting &config_setting,
    std::vector<std::pair<STD, STD>> &match_list, bool &fine_sucess,
    Eigen::Matrix3d &std_rot, Eigen::Vector3d &std_t,
    std::vector<std::pair<STD, STD>> &sucess_match_list,
    std::vector<std::pair<STD, STD>> &unsucess_match_list) {
  sucess_match_list.clear();
  unsucess_match_list.clear();
  int skip_len = (int)(match_list.size() / 200) + 1;
  // old 2
  double dis_threshold = 5.0;
  fine_sucess = false;
  int max_vote = 0;
  std::time_t solve_time = 0;
  std::time_t verify_time = 0;
  std::vector<Eigen::Vector3d> translation_list;
  std::vector<Eigen::Matrix3d> rotation_list;
  for (size_t i = 0; i < match_list.size(); i = i + skip_len) {
    auto single_pair = match_list[i];
    int vote = 0;
    Eigen::Matrix3d test_rot;
    Eigen::Vector3d test_t;
    std::vector<std::pair<STD, STD>> temp_match_list;
    std::vector<std::pair<STD, STD>> temp_unmatch_list;
    triangle_solver(single_pair, test_rot, test_t);
    translation_list.push_back(test_t);
    rotation_list.push_back(test_rot);
  }
  int best_match_number = 0;
  int best_index = -1;
  for (size_t i = 0; i < rotation_list.size(); i++) {
    Eigen::Quaterniond single_q(rotation_list[i]);
    Eigen::Vector3d single_t = translation_list[i];
    int match_number = 0;
    for (size_t j = 0; j < rotation_list.size(); j++) {
      Eigen::Quaterniond match_q(rotation_list[j]);
      Eigen::Vector3d match_t(translation_list[j]);
      // &&(single_t - match_t).norm() < 10
      if (single_q.angularDistance(match_q) < DEG2RAD(10)) {

        match_number++;
      }
    }
    if (match_number > best_match_number) {
      best_match_number = match_number;
      best_index = i;
    }
  }
  // std::cout << "best match number:" << best_match_number << std::endl;
  if (best_match_number >= 5) {
    std_rot = rotation_list[best_index];
    std_t = translation_list[best_index];
    for (auto verify_pair : match_list) {
      Eigen::Vector3d A = verify_pair.first.binary_A_.location_;
      Eigen::Vector3d A_transform = std_rot * A + std_t;
      Eigen::Vector3d B = verify_pair.first.binary_B_.location_;
      Eigen::Vector3d B_transform = std_rot * B + std_t;
      Eigen::Vector3d C = verify_pair.first.binary_C_.location_;
      Eigen::Vector3d C_transform = std_rot * C + std_t;
      double dis_A =
          (A_transform - verify_pair.second.binary_A_.location_).norm();
      double dis_B =
          (B_transform - verify_pair.second.binary_B_.location_).norm();
      double dis_C =
          (C_transform - verify_pair.second.binary_C_.location_).norm();
      if (dis_A < dis_threshold && dis_B < dis_threshold &&
          dis_C < dis_threshold) {
        sucess_match_list.push_back(verify_pair);
      } else {
        unsucess_match_list.push_back(verify_pair);
      }
    }
  }
  if (sucess_match_list.size() >= 5) {
    fine_sucess = true;
  } else {
    fine_sucess = false;
  }
}

void fine_loop_detection_tbb(
    const ConfigSetting &config_setting,
    std::vector<std::pair<STD, STD>> &match_list, bool &fine_sucess,
    Eigen::Matrix3d &std_rot, Eigen::Vector3d &std_t,
    std::vector<std::pair<STD, STD>> &sucess_match_list,
    std::vector<std::pair<STD, STD>> &unsucess_match_list, std::fstream & debug_file) {
  sucess_match_list.clear();
  unsucess_match_list.clear();
  double dis_threshold = 3;
  fine_sucess = false;
  std::time_t solve_time = 0;
  std::time_t verify_time = 0;
  int skip_len = (int)(match_list.size() / 50) + 1;
  int use_size = match_list.size() / skip_len;
  std::vector<size_t> index(use_size);
  std::vector<int> vote_list(use_size);
  for (size_t i = 0; i < index.size(); i++) {
    index[i] = i;
  }
  std::mutex mylock;
  auto t0 = std::chrono::high_resolution_clock::now();
  std::for_each(
      std::execution::par_unseq, index.begin(), index.end(),
      [&](const size_t &i) {
        auto single_pair = match_list[i * skip_len];
        int vote = 0;
        Eigen::Matrix3d test_rot;
        Eigen::Vector3d test_t;
        triangle_solver(single_pair, test_rot, test_t);
        for (size_t j = 0; j < match_list.size(); j++) {
          auto verify_pair = match_list[j];
          Eigen::Vector3d A = verify_pair.first.binary_A_.location_;
          Eigen::Vector3d A_transform = test_rot * A + test_t;
          Eigen::Vector3d B = verify_pair.first.binary_B_.location_;
          Eigen::Vector3d B_transform = test_rot * B + test_t;
          Eigen::Vector3d C = verify_pair.first.binary_C_.location_;
          Eigen::Vector3d C_transform = test_rot * C + test_t;
          double dis_A =
              (A_transform - verify_pair.second.binary_A_.location_).norm();
          double dis_B =
              (B_transform - verify_pair.second.binary_B_.location_).norm();
          double dis_C =
              (C_transform - verify_pair.second.binary_C_.location_).norm();
          if (dis_A < dis_threshold && dis_B < dis_threshold &&
              dis_C < dis_threshold) {
            vote++;
          }
        }
        mylock.lock();
        vote_list[i] = vote;
        mylock.unlock();
      });
  int max_vote_index = 0;
  int max_vote = 0;
  for (size_t i = 0; i < vote_list.size(); i++) {
    if (max_vote < vote_list[i]) {
      max_vote_index = i;
      max_vote = vote_list[i];
    }
  }
  // old 4
//  if (max_vote >= 4) {
if (max_vote >= config_setting.ransac_Rt_thr) {
    fine_sucess = true;
    auto best_pair = match_list[max_vote_index * skip_len];
    int vote = 0;
    Eigen::Matrix3d test_rot;
    Eigen::Vector3d test_t;
    triangle_solver(best_pair, test_rot, test_t);
    std_rot = test_rot;
    std_t = test_t;
    for (size_t j = 0; j < match_list.size(); j++) {
      auto verify_pair = match_list[j];
      Eigen::Vector3d A = verify_pair.first.binary_A_.location_;
      Eigen::Vector3d A_transform = test_rot * A + test_t;
      Eigen::Vector3d B = verify_pair.first.binary_B_.location_;
      Eigen::Vector3d B_transform = test_rot * B + test_t;
      Eigen::Vector3d C = verify_pair.first.binary_C_.location_;
      Eigen::Vector3d C_transform = test_rot * C + test_t;
      double dis_A =
          (A_transform - verify_pair.second.binary_A_.location_).norm();
      double dis_B =
          (B_transform - verify_pair.second.binary_B_.location_).norm();
      double dis_C =
          (C_transform - verify_pair.second.binary_C_.location_).norm();
      if (dis_A < dis_threshold && dis_B < dis_threshold &&
          dis_C < dis_threshold) {
        sucess_match_list.push_back(verify_pair);
      } else {
        unsucess_match_list.push_back(verify_pair);
      }
    }
  } else {
    fine_sucess = false;
  }
  return;
}

double
geometric_verify(const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &source_cloud,
                 const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &target_cloud,
                 const Eigen::Matrix3d &rot, const Eigen::Vector3d &t) {
  pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kd_tree(
      new pcl::KdTreeFLANN<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  for (size_t i = 0; i < target_cloud->size(); i++) {
    pcl::PointXYZ pi;
    pi.x = target_cloud->points[i].x;
    pi.y = target_cloud->points[i].y;
    pi.z = target_cloud->points[i].z;
    input_cloud->push_back(pi);
  }

  kd_tree->setInputCloud(input_cloud);
  // 创建两个向量，分别存放近邻的索引值、近邻的中心距
  std::vector<int> pointIdxNKNSearch(1);
  std::vector<float> pointNKNSquaredDistance(1);
  double useful_match = 0;
  double normal_threshold = 0.1;
  double dis_threshold = 0.3;
  for (size_t i = 0; i < source_cloud->size(); i++) {
    pcl::PointXYZINormal searchPoint = source_cloud->points[i];
    pcl::PointXYZ use_search_point;
    use_search_point.x = searchPoint.x;
    use_search_point.y = searchPoint.y;
    use_search_point.z = searchPoint.z;
    Eigen::Vector3d pi(searchPoint.x, searchPoint.y, searchPoint.z);
    pi = rot * pi + t;
    use_search_point.x = pi[0];
    use_search_point.y = pi[1];
    use_search_point.z = pi[2];
    Eigen::Vector3d ni(searchPoint.normal_x, searchPoint.normal_y,
                       searchPoint.normal_z);
    ni = rot * ni;
    if (kd_tree->nearestKSearch(use_search_point, 1, pointIdxNKNSearch,
                                pointNKNSquaredDistance) > 0) {
      pcl::PointXYZINormal nearstPoint =
          target_cloud->points[pointIdxNKNSearch[0]];
      Eigen::Vector3d tpi(nearstPoint.x, nearstPoint.y, nearstPoint.z);
      Eigen::Vector3d tni(nearstPoint.normal_x, nearstPoint.normal_y,
                          nearstPoint.normal_z);
      Eigen::Vector3d normal_inc = ni - tni;
      Eigen::Vector3d normal_add = ni + tni;
      double point_to_plane = tni.transpose() * (pi - tpi);
      if ((normal_inc.norm() < normal_threshold ||
           normal_add.norm() < normal_threshold) &&
          point_to_plane < dis_threshold) {
        useful_match++;
      }
    }
  }
  return useful_match / source_cloud->size();
}

double
geometric_verify(const ConfigSetting &config_setting,
                 const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &source_cloud,
                 const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &target_cloud,
                 const Eigen::Matrix3d &rot, const Eigen::Vector3d &t) {
  pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kd_tree(
      new pcl::KdTreeFLANN<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  for (size_t i = 0; i < target_cloud->size(); i++) {
    pcl::PointXYZ pi;
    pi.x = target_cloud->points[i].x;
    pi.y = target_cloud->points[i].y;
    pi.z = target_cloud->points[i].z;
    input_cloud->push_back(pi);
  }

  kd_tree->setInputCloud(input_cloud);
  // 创建两个向量，分别存放近邻的索引值、近邻的中心距
  std::vector<int> pointIdxNKNSearch(1);
  std::vector<float> pointNKNSquaredDistance(1);
  double useful_match = 0;
  double normal_threshold = config_setting.normal_threshold_;
  double dis_threshold = config_setting.dis_threshold_;
  for (size_t i = 0; i < source_cloud->size(); i++) {
    pcl::PointXYZINormal searchPoint = source_cloud->points[i];
    pcl::PointXYZ use_search_point;
    use_search_point.x = searchPoint.x;
    use_search_point.y = searchPoint.y;
    use_search_point.z = searchPoint.z;
    Eigen::Vector3d pi(searchPoint.x, searchPoint.y, searchPoint.z);
    pi = rot * pi + t;
    use_search_point.x = pi[0];
    use_search_point.y = pi[1];
    use_search_point.z = pi[2];
    Eigen::Vector3d ni(searchPoint.normal_x, searchPoint.normal_y,
                       searchPoint.normal_z);
    ni = rot * ni;
    if (kd_tree->nearestKSearch(use_search_point, 1, pointIdxNKNSearch,
                                pointNKNSquaredDistance) > 0) {
      pcl::PointXYZINormal nearstPoint =
          target_cloud->points[pointIdxNKNSearch[0]];
      Eigen::Vector3d tpi(nearstPoint.x, nearstPoint.y, nearstPoint.z);
      Eigen::Vector3d tni(nearstPoint.normal_x, nearstPoint.normal_y,
                          nearstPoint.normal_z);
      Eigen::Vector3d normal_inc = ni - tni;
      Eigen::Vector3d normal_add = ni + tni;
      double point_to_plane = fabs(tni.transpose() * (pi - tpi));
      if ((normal_inc.norm() < normal_threshold ||
           normal_add.norm() < normal_threshold) &&
          point_to_plane < dis_threshold) {
        useful_match++;
      }
    }
  }
  return useful_match / source_cloud->size();
}

struct PlaneSolver {
  PlaneSolver(Eigen::Vector3d curr_point_, Eigen::Vector3d curr_normal_,
              Eigen::Vector3d target_point_, Eigen::Vector3d target_normal_)
      : curr_point(curr_point_), curr_normal(curr_normal_),
        target_point(target_point_), target_normal(target_normal_){};
  template <typename T>
  bool operator()(const T *q, const T *t, T *residual) const {
    Eigen::Quaternion<T> q_w_curr{q[3], q[0], q[1], q[2]};
    Eigen::Matrix<T, 3, 1> t_w_curr{t[0], t[1], t[2]};
    Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()),
                              T(curr_point.z())};
    Eigen::Matrix<T, 3, 1> point_w;
    point_w = q_w_curr * cp + t_w_curr;
    Eigen::Matrix<T, 3, 1> point_target(
        T(target_point.x()), T(target_point.y()), T(target_point.z()));
    Eigen::Matrix<T, 3, 1> norm(T(target_normal.x()), T(target_normal.y()),
                                T(target_normal.z()));
    residual[0] = norm.dot(point_w - point_target);
    return true;
  }

  static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_,
                                     const Eigen::Vector3d curr_normal_,
                                     Eigen::Vector3d target_point_,
                                     Eigen::Vector3d target_normal_) {
    return (
        new ceres::AutoDiffCostFunction<PlaneSolver, 1, 4, 3>(new PlaneSolver(
            curr_point_, curr_normal_, target_point_, target_normal_)));
  }

  Eigen::Vector3d curr_point;
  Eigen::Vector3d curr_normal;
  Eigen::Vector3d target_point;
  Eigen::Vector3d target_normal;
};

double geometric_icp(const ConfigSetting &config_setting,
    const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &cloud,
    const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &target_cloud,
    Eigen::Matrix3d &rot, Eigen::Vector3d &t) {
  pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kd_tree(
      new pcl::KdTreeFLANN<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  for (size_t i = 0; i < target_cloud->size(); i++) {
    pcl::PointXYZ pi;
    pi.x = target_cloud->points[i].x;
    pi.y = target_cloud->points[i].y;
    pi.z = target_cloud->points[i].z;
    input_cloud->push_back(pi);
  }
  kd_tree->setInputCloud(input_cloud);
  Eigen::Quaterniond q(rot.cast<double>());
  ceres::LocalParameterization *q_parameterization =
      new ceres::EigenQuaternionParameterization();
  ceres::Problem problem;
  ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
  double para_q[4] = {q.x(), q.y(), q.z(), q.w()};
  double para_t[3] = {t(0), t(1), t(2)};
  problem.AddParameterBlock(para_q, 4, q_parameterization);
  problem.AddParameterBlock(para_t, 3);
  Eigen::Map<Eigen::Quaterniond> q_last_curr(para_q);
  Eigen::Map<Eigen::Vector3d> t_last_curr(para_t);
  // 创建两个向量，分别存放近邻的索引值、近邻的中心距
  std::vector<int> pointIdxNKNSearch(1);
  std::vector<float> pointNKNSquaredDistance(1);
  double useful_match = 0;
//  double normal_threshold = 0.1;
//  double dis_threshold = 0.2;
  double normal_threshold = config_setting.normal_threshold_;
  double dis_threshold = config_setting.dis_threshold_;
  for (size_t i = 0; i < cloud->size(); i++) {
    pcl::PointXYZINormal searchPoint = cloud->points[i];
    Eigen::Vector3d pi(searchPoint.x, searchPoint.y, searchPoint.z);
    pi = rot * pi + t;
    pcl::PointXYZ use_search_point;
    use_search_point.x = pi[0];
    use_search_point.y = pi[1];
    use_search_point.z = pi[2];
    Eigen::Vector3d ni(searchPoint.normal_x, searchPoint.normal_y,
                       searchPoint.normal_z);
    ni = rot * ni;
    if (kd_tree->nearestKSearch(use_search_point, 1, pointIdxNKNSearch,
                                pointNKNSquaredDistance) > 0) {
      pcl::PointXYZINormal nearstPoint =
          target_cloud->points[pointIdxNKNSearch[0]];
      Eigen::Vector3d tpi(nearstPoint.x, nearstPoint.y, nearstPoint.z);
      Eigen::Vector3d tni(nearstPoint.normal_x, nearstPoint.normal_y,
                          nearstPoint.normal_z);
      Eigen::Vector3d normal_inc = ni - tni;
      Eigen::Vector3d normal_add = ni + tni;
      double point_to_plane = tni.transpose() * (pi - tpi);
      if ((normal_inc.norm() < normal_threshold ||
           normal_add.norm() < normal_threshold) &&
          point_to_plane < dis_threshold) {
        useful_match++;
        ceres::CostFunction *cost_function;
        Eigen::Vector3d curr_point(cloud->points[i].x, cloud->points[i].y,
                                   cloud->points[i].z);
        Eigen::Vector3d curr_normal(cloud->points[i].normal_x,
                                    cloud->points[i].normal_y,
                                    cloud->points[i].normal_z);

        cost_function = PlaneSolver::Create(curr_point, curr_normal, tpi, tni);
        problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
      }
    }
  }
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.max_num_iterations = 4;
  options.minimizer_progress_to_stdout = false;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  Eigen::Quaterniond q_opt(para_q[3], para_q[0], para_q[1], para_q[2]);
  // std::cout << summary.BriefReport() << std::endl;
  rot = q_opt.toRotationMatrix();
  t << t_last_curr(0), t_last_curr(1), t_last_curr(2);
  std::cout << "useful match for plane icp:" << useful_match << std::endl;
  return useful_match / cloud->size();
}

void add_STD(std::unordered_map<STD_LOC, std::vector<STD>> &descriptor_map,
             std::vector<STD> &STD_list) {
  for (auto single_std : STD_list) {
    STD_LOC position;
    position.x = (int)(single_std.triangle_[0] + 0.5);
    position.y = (int)(single_std.triangle_[1] + 0.5);
    position.z = (int)(single_std.triangle_[2] + 0.5);
    position.a = (int)(single_std.angle_[0]);
    position.b = (int)(single_std.angle_[1]);
    position.c = (int)(single_std.angle_[2]);
    auto iter = descriptor_map.find(position);
    // single_std.score_frame_.push_back(single_std.frame_number_);
    if (iter != descriptor_map.end()) {
      descriptor_map[position].push_back(single_std);
    } else {
      std::vector<STD> descriptor_list;
      descriptor_list.push_back(single_std);
      descriptor_map[position] = descriptor_list;
    }
  }
}

void publish_binary(const std::vector<BinaryDescriptor> &binary_list,
                    const Eigen::Vector3d &text_color,
                    const std::string &text_ns,
                    const ros::Publisher &text_publisher) {
  visualization_msgs::MarkerArray text_array;
  visualization_msgs::Marker text;
  text.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
  text.action = visualization_msgs::Marker::ADD;
  text.ns = text_ns;
  text.color.a = 0.8; // Don't forget to set the alpha!
  text.scale.z = 0.08;
  text.pose.orientation.w = 1.0;
  text.header.frame_id = "camera_init";
  for (size_t i = 0; i < binary_list.size(); i++) {
    text.pose.position.x = binary_list[i].location_[0];
    text.pose.position.y = binary_list[i].location_[1];
    text.pose.position.z = binary_list[i].location_[2];
    std::ostringstream str;
    str << std::to_string((int)(binary_list[i].summary_));
    text.text = str.str();
    text.scale.x = 0.5;
    text.scale.y = 0.5;
    text.scale.z = 0.5;
    text.color.r = text_color[0];
    text.color.g = text_color[1];
    text.color.b = text_color[2];
    text.color.a = 1;
    text.id++;
    text_array.markers.push_back(text);
  }
  for (int i = 1; i < 100; i++) {
    text.color.a = 0;
    text.id++;
    text_array.markers.push_back(text);
  }
  text_publisher.publish(text_array);
  return;
}

void publish_std_list(const std::vector<STD> &std_list,
                      const ros::Publisher &std_publisher) {
  // publish descriptor
  visualization_msgs::MarkerArray ma_line;
  visualization_msgs::Marker m_line;
  m_line.type = visualization_msgs::Marker::LINE_LIST;
  m_line.action = visualization_msgs::Marker::ADD;
  m_line.ns = "std";
  // Don't forget to set the alpha!
  m_line.scale.x = 0.25;
  m_line.pose.orientation.w = 1.0;
  m_line.header.frame_id = "camera_init";
  m_line.id = 0;
  m_line.points.clear();
  m_line.color.r = 0;
  m_line.color.g = 1;
  m_line.color.b = 0;
  m_line.color.a = 0.8;
  for (auto var : std_list) {
    geometry_msgs::Point p;
    p.x = var.binary_A_.location_[0];
    p.y = var.binary_A_.location_[1];
    p.z = var.binary_A_.location_[2];
    m_line.points.push_back(p);
    p.x = var.binary_B_.location_[0];
    p.y = var.binary_B_.location_[1];
    p.z = var.binary_B_.location_[2];
    m_line.points.push_back(p);
    ma_line.markers.push_back(m_line);
    m_line.id++;
    m_line.points.clear();
    p.x = var.binary_C_.location_[0];
    p.y = var.binary_C_.location_[1];
    p.z = var.binary_C_.location_[2];
    m_line.points.push_back(p);
    p.x = var.binary_B_.location_[0];
    p.y = var.binary_B_.location_[1];
    p.z = var.binary_B_.location_[2];
    m_line.points.push_back(p);
    ma_line.markers.push_back(m_line);
    m_line.id++;
    m_line.points.clear();
    p.x = var.binary_C_.location_[0];
    p.y = var.binary_C_.location_[1];
    p.z = var.binary_C_.location_[2];
    m_line.points.push_back(p);
    p.x = var.binary_A_.location_[0];
    p.y = var.binary_A_.location_[1];
    p.z = var.binary_A_.location_[2];
    m_line.points.push_back(p);
    ma_line.markers.push_back(m_line);
    m_line.id++;
    m_line.points.clear();
  }
  for (int j = 0; j < 1000 * 3; j++) {
    m_line.color.a = 0.00;
    ma_line.markers.push_back(m_line);
    m_line.id++;
  }
  std_publisher.publish(ma_line);
  m_line.id = 0;
  ma_line.markers.clear();
}

void publish_std(const std::vector<std::pair<STD, STD>> &match_std_list,
                 const ros::Publisher &std_publisher) {
  // publish descriptor
  // bool transform_enable = true;
  visualization_msgs::MarkerArray ma_line;
  visualization_msgs::Marker m_line;
  m_line.type = visualization_msgs::Marker::LINE_LIST;
  m_line.action = visualization_msgs::Marker::ADD;
  m_line.ns = "lines";
  // Don't forget to set the alpha!
  m_line.scale.x = 0.25;
  m_line.pose.orientation.w = 1.0;
  m_line.header.frame_id = "camera_init";
  m_line.id = 0;
  int max_pub_cnt = 1;
  for (auto var : match_std_list) {
    if (max_pub_cnt > 100) {
      break;
    }
    max_pub_cnt++;
    m_line.color.a = 0.8;
    m_line.points.clear();
    m_line.color.r = 0;
    m_line.color.g = 1;
    m_line.color.b = 0;
    geometry_msgs::Point p;
    p.x = var.second.binary_A_.location_[0];
    p.y = var.second.binary_A_.location_[1];
    p.z = var.second.binary_A_.location_[2];
    Eigen::Vector3d t_p;
    t_p << p.x, p.y, p.z;
    // if (transform_enable)
    //   t_p = rotation.inverse() * (t_p - translation);
    p.x = t_p[0];
    p.y = t_p[1];
    p.z = t_p[2];
    m_line.points.push_back(p);
    p.x = var.second.binary_B_.location_[0];
    p.y = var.second.binary_B_.location_[1];
    p.z = var.second.binary_B_.location_[2];
    t_p << p.x, p.y, p.z;
    // t_p = rotation.inverse() * (t_p - translation);
    p.x = t_p[0];
    p.y = t_p[1];
    p.z = t_p[2];
    m_line.points.push_back(p);
    ma_line.markers.push_back(m_line);
    m_line.id++;
    m_line.points.clear();
    p.x = var.second.binary_C_.location_[0];
    p.y = var.second.binary_C_.location_[1];
    p.z = var.second.binary_C_.location_[2];
    t_p << p.x, p.y, p.z;
    // t_p = rotation.inverse() * (t_p - translation);
    p.x = t_p[0];
    p.y = t_p[1];
    p.z = t_p[2];
    m_line.points.push_back(p);
    p.x = var.second.binary_B_.location_[0];
    p.y = var.second.binary_B_.location_[1];
    p.z = var.second.binary_B_.location_[2];
    t_p << p.x, p.y, p.z;
    // t_p = rotation.inverse() * (t_p - translation);
    p.x = t_p[0];
    p.y = t_p[1];
    p.z = t_p[2];
    m_line.points.push_back(p);
    ma_line.markers.push_back(m_line);
    m_line.id++;
    m_line.points.clear();
    p.x = var.second.binary_C_.location_[0];
    p.y = var.second.binary_C_.location_[1];
    p.z = var.second.binary_C_.location_[2];
    t_p << p.x, p.y, p.z;
    // t_p = rotation.inverse() * (t_p - translation);
    p.x = t_p[0];
    p.y = t_p[1];
    p.z = t_p[2];
    m_line.points.push_back(p);
    p.x = var.second.binary_A_.location_[0];
    p.y = var.second.binary_A_.location_[1];
    p.z = var.second.binary_A_.location_[2];
    t_p << p.x, p.y, p.z;
    // t_p = rotation.inverse() * (t_p - translation);
    p.x = t_p[0];
    p.y = t_p[1];
    p.z = t_p[2];
    m_line.points.push_back(p);
    ma_line.markers.push_back(m_line);
    m_line.id++;
    m_line.points.clear();
    // another
    m_line.points.clear();
    m_line.color.r = 1;
    m_line.color.g = 1;
    m_line.color.b = 1;
    p.x = var.first.binary_A_.location_[0];
    p.y = var.first.binary_A_.location_[1];
    p.z = var.first.binary_A_.location_[2];
    m_line.points.push_back(p);
    p.x = var.first.binary_B_.location_[0];
    p.y = var.first.binary_B_.location_[1];
    p.z = var.first.binary_B_.location_[2];
    m_line.points.push_back(p);
    ma_line.markers.push_back(m_line);
    m_line.id++;
    m_line.points.clear();
    p.x = var.first.binary_C_.location_[0];
    p.y = var.first.binary_C_.location_[1];
    p.z = var.first.binary_C_.location_[2];
    m_line.points.push_back(p);
    p.x = var.first.binary_B_.location_[0];
    p.y = var.first.binary_B_.location_[1];
    p.z = var.first.binary_B_.location_[2];
    m_line.points.push_back(p);
    ma_line.markers.push_back(m_line);
    m_line.id++;
    m_line.points.clear();
    p.x = var.first.binary_C_.location_[0];
    p.y = var.first.binary_C_.location_[1];
    p.z = var.first.binary_C_.location_[2];
    m_line.points.push_back(p);
    p.x = var.first.binary_A_.location_[0];
    p.y = var.first.binary_A_.location_[1];
    p.z = var.first.binary_A_.location_[2];
    m_line.points.push_back(p);
    ma_line.markers.push_back(m_line);
    m_line.id++;
    m_line.points.clear();
    // debug
    // std_publisher.publish(ma_line);
    // std::cout << "var first: " << var.first.triangle_.transpose()
    //           << " , var second: " << var.second.triangle_.transpose()
    //           << std::endl;
    // getchar();
  }
  for (int j = 0; j < 100 * 6; j++) {
    m_line.color.a = 0.00;
    ma_line.markers.push_back(m_line);
    m_line.id++;
  }
  std_publisher.publish(ma_line);
  m_line.id = 0;
  ma_line.markers.clear();
}

double
calc_triangle_dis(const std::vector<std::pair<STD, STD>> &match_std_list) {
  double mean_triangle_dis = 0;
  for (auto var : match_std_list) {
    mean_triangle_dis += (var.first.triangle_ - var.second.triangle_).norm() /
                         var.first.triangle_.norm();
  }
  if (match_std_list.size() > 0) {
    mean_triangle_dis = mean_triangle_dis / match_std_list.size();
  } else {
    mean_triangle_dis = -1;
  }
  return mean_triangle_dis;
}

double
calc_binary_similaity(const std::vector<std::pair<STD, STD>> &match_std_list) {
  double mean_binary_similarity = 0;
  for (auto var : match_std_list) {
    mean_binary_similarity +=
        (binary_similarity(var.first.binary_A_, var.second.binary_A_) +
         binary_similarity(var.first.binary_B_, var.second.binary_B_) +
         binary_similarity(var.first.binary_C_, var.second.binary_C_)) /
        3;
  }
  if (match_std_list.size() > 0) {
    mean_binary_similarity = mean_binary_similarity / match_std_list.size();
  } else {
    mean_binary_similarity = -1;
  }
  return mean_binary_similarity;
}

double calc_overlap(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud1,
                    const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud2,
                    double dis_threshold) {
  int point_kip = 2;
  double match_num = 0;
  pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kd_tree(
      new pcl::KdTreeFLANN<pcl::PointXYZI>);
  kd_tree->setInputCloud(cloud2);
  std::vector<int> pointIdxNKNSearch(1);
  std::vector<float> pointNKNSquaredDistance(1);
  for (size_t i = 0; i < cloud1->size(); i += point_kip) {
    pcl::PointXYZI searchPoint = cloud1->points[i];
    if (kd_tree->nearestKSearch(searchPoint, 1, pointIdxNKNSearch,
                                pointNKNSquaredDistance) > 0) {
      if (pointNKNSquaredDistance[0] < dis_threshold * dis_threshold) {
        match_num++;
      }
    }
  }
  // std::cout << "cloud1 size:" << cloud1->size()
  //           << " cloud2 size: " << cloud2->size() << " match size:" <<
  //           match_num
  //           << std::endl;
  double overlap =
      2 * match_num * point_kip / (cloud1->size() + cloud2->size());
  return overlap;
}

double calc_overlap(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud1,
                    const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud2,
                    double dis_threshold, int skip_num) {
  double match_num = 0;
  pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kd_tree(
      new pcl::KdTreeFLANN<pcl::PointXYZI>);
  kd_tree->setInputCloud(cloud2);
  std::vector<int> pointIdxNKNSearch(1);
  std::vector<float> pointNKNSquaredDistance(1);
  for (size_t i = 0; i < cloud1->size(); i += skip_num) {
    pcl::PointXYZI searchPoint = cloud1->points[i];
    if (kd_tree->nearestKSearch(searchPoint, 1, pointIdxNKNSearch,
                                pointNKNSquaredDistance) > 0) {
      if (pointNKNSquaredDistance[0] < dis_threshold * dis_threshold) {
        match_num++;
      }
    }
  }
  // std::cout << "cloud1 size:" << cloud1->size()
  //           << " cloud2 size: " << cloud2->size() << " match size:" <<
  //           match_num
  //           << std::endl;
  double overlap =
      (2 * match_num * skip_num) / (cloud1->size() + cloud2->size());
  return overlap;
}

void CalcQuation(const Eigen::Vector3d &vec, const int axis,
                 geometry_msgs::Quaternion &q) {
  Eigen::Vector3d x_body = vec;
  Eigen::Vector3d y_body(1, 1, 0);
  if (x_body(2) != 0) {
    y_body(2) = -(y_body(0) * x_body(0) + y_body(1) * x_body(1)) / x_body(2);
  } else {
    if (x_body(1) != 0) {
      y_body(1) = -(y_body(0) * x_body(0)) / x_body(1);
    } else {
      y_body(0) = 0;
    }
  }
  y_body.normalize();
  Eigen::Vector3d z_body = x_body.cross(y_body);
  Eigen::Matrix3d rot;

  rot << x_body(0), x_body(1), x_body(2), y_body(0), y_body(1), y_body(2),
      z_body(0), z_body(1), z_body(2);
  Eigen::Matrix3d rotation = rot.transpose();
  if (axis == 2) {
    Eigen::Matrix3d rot_inc;
    rot_inc << 0, 0, 1, 0, 1, 0, -1, 0, 0;
    rotation = rotation * rot_inc;
  }
  Eigen::Quaterniond eq(rotation);
  q.w = eq.w();
  q.x = eq.x();
  q.y = eq.y();
  q.z = eq.z();
}

void pubPlane(const ros::Publisher &plane_pub, const std::string plane_ns,
              const int plane_id, const pcl::PointXYZINormal normal_p,
              const float radius, const Eigen::Vector3d rgb) {
  visualization_msgs::Marker plane;
  plane.header.frame_id = "camera_init";
  plane.header.stamp = ros::Time();
  plane.ns = plane_ns;
  plane.id = plane_id;
  plane.type = visualization_msgs::Marker::CUBE;
  plane.action = visualization_msgs::Marker::ADD;
  plane.pose.position.x = normal_p.x;
  plane.pose.position.y = normal_p.y;
  plane.pose.position.z = normal_p.z;
  geometry_msgs::Quaternion q;
  Eigen::Vector3d normal_vec(normal_p.normal_x, normal_p.normal_y,
                             normal_p.normal_z);
  CalcQuation(normal_vec, 2, q);
  plane.pose.orientation = q;
  plane.scale.x = 2.0 * radius;
  plane.scale.y = 2.0 * radius;
  plane.scale.z = 0.1;
  plane.color.a = 0.5;
  plane.color.r = fabs(rgb(0));
  plane.color.g = fabs(rgb(1));
  plane.color.b = fabs(rgb(2));
  plane.lifetime = ros::Duration();
  plane_pub.publish(plane);
}

pcl::PointXYZI vec2point(const Eigen::Vector3d &vec) {
  pcl::PointXYZI pi;
  pi.x = vec[0];
  pi.y = vec[1];
  pi.z = vec[2];
  return pi;
}
Eigen::Vector3d point2vec(const pcl::PointXYZI &pi) {
  return Eigen::Vector3d(pi.x, pi.y, pi.z);
}

Eigen::Vector3d normal2vec(const pcl::PointXYZINormal &pi) {
  Eigen::Vector3d vec(pi.normal_x, pi.normal_y, pi.normal_z);
  return vec;
}

double time_inc(std::chrono::_V2::system_clock::time_point &t_end,
                std::chrono::_V2::system_clock::time_point &t_begin) {
  return std::chrono::duration_cast<std::chrono::duration<double>>(t_end -
                                                                   t_begin)
             .count() *
         1000;
}

void save_an_array(const BinaryDescriptor& bdes,pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_to_store){
  pcl::PointXYZ pt_tmp;
  for (size_t i = 0; i < bdes.occupy_array_.size(); i+=3) {
    pt_tmp.x = bdes.occupy_array_[i];
    if (i+1 < bdes.occupy_array_.size())
       pt_tmp.y = bdes.occupy_array_[i+1];
    else
       pt_tmp.y = -100;
    if (i+2 < bdes.occupy_array_.size())
       pt_tmp.z = bdes.occupy_array_[i+2];
    else
       pt_tmp.z = -100;
    cloud_to_store->push_back(pt_tmp);
  }
}

void save_descriptor(const std::vector<STD> &current_descriptor,
                     pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_to_store) {
  for (auto descrip : current_descriptor) {
    pcl::PointXYZ pt_tmp;
    pt_tmp.x = -1010.1; pt_tmp.y = descrip.frame_number_; pt_tmp.z = 0;  //-1010.1 index denote a new descriptor
    cloud_to_store->points.push_back(pt_tmp);
    pt_tmp.x = descrip.triangle_[0]; pt_tmp.y = descrip.triangle_[1]; pt_tmp.z = descrip.triangle_[2];
    cloud_to_store->points.push_back(pt_tmp);
    pt_tmp.x = descrip.angle_[0]; pt_tmp.y = descrip.angle_[1]; pt_tmp.z = descrip.angle_[2];
    cloud_to_store->points.push_back(pt_tmp);
    pt_tmp.x = descrip.center_[0]; pt_tmp.y = descrip.center_[1]; pt_tmp.z = descrip.center_[2];
    cloud_to_store->points.push_back(pt_tmp);

    pt_tmp.x = descrip.binary_A_.location_[0]; pt_tmp.y = descrip.binary_A_.location_[1]; pt_tmp.z = descrip.binary_A_.location_[2];
    cloud_to_store->points.push_back(pt_tmp);
    pt_tmp.x = descrip.binary_B_.location_[0]; pt_tmp.y = descrip.binary_B_.location_[1]; pt_tmp.z = descrip.binary_B_.location_[2];
    cloud_to_store->points.push_back(pt_tmp);
    pt_tmp.x = descrip.binary_C_.location_[0]; pt_tmp.y = descrip.binary_C_.location_[1]; pt_tmp.z = descrip.binary_C_.location_[2];
    cloud_to_store->points.push_back(pt_tmp);
    pt_tmp.x = float(descrip.binary_A_.summary_); pt_tmp.y = float(descrip.binary_B_.summary_); pt_tmp.z = float(descrip.binary_C_.summary_);
    cloud_to_store->points.push_back(pt_tmp);
    pt_tmp.x = descrip.binary_A_.occupy_array_.size(); pt_tmp.y = descrip.binary_B_.occupy_array_.size(); pt_tmp.z = descrip.binary_C_.occupy_array_.size();
    cloud_to_store->points.push_back(pt_tmp);
//    std::cout << "occupy_array size: " << pt_tmp.x << ", " << pt_tmp.y << ", " << pt_tmp.z <<std::endl;
    save_an_array(descrip.binary_A_, cloud_to_store);
    save_an_array(descrip.binary_B_, cloud_to_store);
    save_an_array(descrip.binary_C_, cloud_to_store);

  }
}

void load_array(std::vector<bool> &occupy_array_, const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_to_load, const int i)
{
    int j = 0;
    occupy_array_[j*3] = cloud_to_load->points[i+j].x;
    occupy_array_[j*3+1] = cloud_to_load->points[i+j].y;
    occupy_array_[j*3+2] = cloud_to_load->points[i+j].z;
    j++;
    occupy_array_[j*3] = cloud_to_load->points[i+j].x;
    occupy_array_[j*3+1] = cloud_to_load->points[i+j].y;
    occupy_array_[j*3+2] = cloud_to_load->points[i+j].z;
    j++;
    occupy_array_[j*3] = cloud_to_load->points[i+j].x;
    occupy_array_[j*3+1] = cloud_to_load->points[i+j].y;
    occupy_array_[j*3+2] = cloud_to_load->points[i+j].z;
    j++;
    occupy_array_[j*3] = cloud_to_load->points[i+j].x;
    occupy_array_[j*3+1] = cloud_to_load->points[i+j].y;
    occupy_array_[j*3+2] = cloud_to_load->points[i+j].z;
    j++;
    occupy_array_[j*3] = cloud_to_load->points[i+j].x;
    occupy_array_[j*3+1] = cloud_to_load->points[i+j].y;
    occupy_array_[j*3+2] = cloud_to_load->points[i+j].z;
    j++;
    occupy_array_[j*3] = cloud_to_load->points[i+j].x;
    occupy_array_[j*3+1] = cloud_to_load->points[i+j].y;
    occupy_array_[j*3+2] = cloud_to_load->points[i+j].z;
    j++;
    occupy_array_[j*3] = cloud_to_load->points[i+j].x;
    occupy_array_[j*3+1] = cloud_to_load->points[i+j].y;
    occupy_array_[j*3+2] = cloud_to_load->points[i+j].z;
    j++;
    occupy_array_[j*3] = cloud_to_load->points[i+j].x;
    occupy_array_[j*3+1] = cloud_to_load->points[i+j].y;
    occupy_array_[j*3+2] = cloud_to_load->points[i+j].z;
    j++;
    occupy_array_[j*3] = cloud_to_load->points[i+j].x;
    occupy_array_[j*3+1] = cloud_to_load->points[i+j].y;
    occupy_array_[j*3+2] = cloud_to_load->points[i+j].z;
    j++;
    occupy_array_[j*3] = cloud_to_load->points[i+j].x;
    occupy_array_[j*3+1] = cloud_to_load->points[i+j].y;
    occupy_array_[j*3+2] = cloud_to_load->points[i+j].z; //30
        j++;
    occupy_array_[j*3] = cloud_to_load->points[i+j].x;
    occupy_array_[j*3+1] = cloud_to_load->points[i+j].y;
    occupy_array_[j*3+2] = cloud_to_load->points[i+j].z;
        j++;
    occupy_array_[j*3] = cloud_to_load->points[i+j].x;
    occupy_array_[j*3+1] = cloud_to_load->points[i+j].y;
    occupy_array_[j*3+2] = cloud_to_load->points[i+j].z;
        j++;
    occupy_array_[j*3] = cloud_to_load->points[i+j].x;
    occupy_array_[j*3+1] = cloud_to_load->points[i+j].y;
    occupy_array_[j*3+2] = cloud_to_load->points[i+j].z;
        j++;
    occupy_array_[j*3] = cloud_to_load->points[i+j].x;
    occupy_array_[j*3+1] = cloud_to_load->points[i+j].y;
    occupy_array_[j*3+2] = cloud_to_load->points[i+j].z;
        j++;
    occupy_array_[j*3] = cloud_to_load->points[i+j].x;
    occupy_array_[j*3+1] = cloud_to_load->points[i+j].y;
    occupy_array_[j*3+2] = cloud_to_load->points[i+j].z;
        j++;
    occupy_array_[j*3] = cloud_to_load->points[i+j].x;
    occupy_array_[j*3+1] = cloud_to_load->points[i+j].y;
//    occupy_array_[j*3+2] = cloud_to_load->points[i+j].z;
}

void load_descriptor(std::unordered_map<STD_LOC, std::vector<STD>>  &feat_map,
                     const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_to_load, int &index_max) {
  if (cloud_to_load->points.size() == 0)
    return;
//  std::vector<Descriptor> descriptor_list;
//  descriptor_list.reserve(1000);
  STD descrip;
  int occupy_array_size = 47;
  descrip.binary_A_.occupy_array_.resize(occupy_array_size);
  descrip.binary_B_.occupy_array_.resize(occupy_array_size);
  descrip.binary_C_.occupy_array_.resize(occupy_array_size);

  size_t i = 0;
  feat_map.reserve(1000000);
  while(i < cloud_to_load->points.size()){
    //if (i%1000 == 0) std::cout << i/1000 << "%" << std::endl;
    if( fabs(cloud_to_load->points[i].x - (-1010.1)) < 0.01){
      descrip.frame_number_ = cloud_to_load->points[i].y;
      index_max = descrip.frame_number_>index_max?descrip.frame_number_:index_max;
      i++;
      descrip.triangle_[0] = cloud_to_load->points[i].x;
      descrip.triangle_[1] = cloud_to_load->points[i].y;
      descrip.triangle_[2] = cloud_to_load->points[i].z;
      i++;
      descrip.angle_[0] = cloud_to_load->points[i].x;
      descrip.angle_[1] = cloud_to_load->points[i].y;
      descrip.angle_[2] = cloud_to_load->points[i].z;
      i++;
      descrip.center_[0] = cloud_to_load->points[i].x;
      descrip.center_[1] = cloud_to_load->points[i].y;
      descrip.center_[2] = cloud_to_load->points[i].z;
      i++;
      descrip.binary_A_.location_[0] = cloud_to_load->points[i].x;
      descrip.binary_A_.location_[1] = cloud_to_load->points[i].y;
      descrip.binary_A_.location_[2] = cloud_to_load->points[i].z;
      i++;
      descrip.binary_B_.location_[0] = cloud_to_load->points[i].x;
      descrip.binary_B_.location_[1] = cloud_to_load->points[i].y;
      descrip.binary_B_.location_[2] = cloud_to_load->points[i].z;
      i++;
      descrip.binary_C_.location_[0] = cloud_to_load->points[i].x;
      descrip.binary_C_.location_[1] = cloud_to_load->points[i].y;
      descrip.binary_C_.location_[2] = cloud_to_load->points[i].z;
      i++;
      descrip.binary_A_.summary_ = cloud_to_load->points[i].x;
      descrip.binary_B_.summary_ = cloud_to_load->points[i].y;
      descrip.binary_C_.summary_ = cloud_to_load->points[i].z;
      i++;
      int bin_A_size = ceil(cloud_to_load->points[i].x/3);
      int bin_B_size = ceil(cloud_to_load->points[i].y/3);
      int bin_C_size = ceil(cloud_to_load->points[i].z/3);
      i++;

      load_array(descrip.binary_A_.occupy_array_, cloud_to_load, i);
      i+=bin_A_size;
      load_array(descrip.binary_B_.occupy_array_, cloud_to_load, i);
      i+=bin_B_size;
      load_array(descrip.binary_C_.occupy_array_, cloud_to_load, i);
      i+=bin_C_size;

      STD_LOC position;
      position.x = (int)(descrip.triangle_[0] + 0.5);
      position.y = (int)(descrip.triangle_[1] + 0.5);
      position.z = (int)(descrip.triangle_[2] + 0.5);
      position.a = (int)(descrip.angle_[0]);
      position.b = (int)(descrip.angle_[1]);
      position.c = (int)(descrip.angle_[2]);
      auto iter = feat_map.find(position);
      if (iter != feat_map.end()) {
        feat_map[position].push_back(descrip);
      } else {
        std::vector<STD> descriptor_list;
        descriptor_list.reserve(100);
        descriptor_list.push_back(descrip);
        feat_map[position] = descriptor_list;
//        std::cout << feat_map.size() << " ";
      }
    }
  }
}
