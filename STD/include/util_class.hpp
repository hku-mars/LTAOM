#ifndef UTIL_CALSS_HPP
#define UTIL_CALSS_HPP
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <cv_bridge/cv_bridge.h>
#include <execution>
#include <opencv2/opencv.hpp>
#include <pcl/common/io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ros/ros.h>
#include <stdio.h>
#include <string>
#include <unordered_map>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#define HASH_P 116101
#define MAX_N 10000000000
#define max_layer 0

typedef struct ConfigSetting {
  double voxel_size = 1.0;
  int alternative_num = 100;
  double ds_size = 0.5;
  int max_frame = 100;
  int sub_frame_num = 50;
  int useful_corner_num = 30;
  int VoxelPointSize = 10;
  double planer_threshold = 0.01;
  float icp_threshold = 0.5;
  double descriptor_near_num = 10;
  double descriptor_min_len = 1;
  double descriptor_max_len = 10;
  double max_count_threshold = 10;
  double max_constraint_dis = 3.0;
  float triangle_resolution = 0.2;
} ConfigSetting;

//
typedef struct BinaryDescriptor {
  std::vector<bool> occupy_array;
  unsigned char summary = 0;
  Eigen::Vector3d location;
} BinaryDescriptor;

// 1kb,12.8
typedef struct Descriptor {
  Eigen::Vector3d triangle;
  Eigen::Vector3d angle;
  Eigen::Vector3d A;
  Eigen::Vector3d B;
  Eigen::Vector3d C;
  unsigned char count_A;
  unsigned char count_B;
  unsigned char count_C;
  Eigen::Vector3d center;
  float triangle_scale;
  unsigned short frame_number;
  std::vector<unsigned short> score_frame;
  std::vector<Eigen::Matrix3d> position_list;
  BinaryDescriptor binary_A;
  BinaryDescriptor binary_B;
  BinaryDescriptor binary_C;
} Descriptor;

typedef struct Plane {
  // pcl::PointCloud<pcl::PointXYZI> cloud;
  pcl::PointXYZINormal p_center;
  Eigen::Vector3d center;
  Eigen::Vector3d normal;
  Eigen::Matrix3d covariance;
  float radius = 0;
  float min_eigen_value = 1;
  float d = 0;
  int id = 0;
  int sub_plane_num = 0;
  int points_size = 0;
  bool is_plane = false;
} Plane;

typedef struct MatchTriangleList {
  std::vector<std::pair<Descriptor, Descriptor>> match_list;
  int match_frame;
  double mean_dis;
} MatchTriangleList;

typedef struct key_point {
  pcl::PointXYZINormal p;
  double count;
} key_point;

bool GreaterSort(key_point a, key_point b) { return (a.count > b.count); }

bool BinaryGreaterSort(BinaryDescriptor a, BinaryDescriptor b) {
  return (a.summary > b.summary);
}

bool CompGreater(Plane *plane1, Plane *plane2) {
  return plane1->sub_plane_num > plane2->sub_plane_num;
}
double time_inc(std::chrono::_V2::system_clock::time_point &t_end,
                std::chrono::_V2::system_clock::time_point &t_begin) {
  return std::chrono::duration_cast<std::chrono::duration<double>>(t_end -
                                                                   t_begin)
             .count() *
         1000;
}

class VOXEL_LOC {
public:
  int64_t x, y, z;

  VOXEL_LOC(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0)
      : x(vx), y(vy), z(vz) {}

  bool operator==(const VOXEL_LOC &other) const {
    return (x == other.x && y == other.y && z == other.z);
  }
};

class DESCRIPTOR_LOC {
public:
  int64_t x, y, z, a, b, c;

  DESCRIPTOR_LOC(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0, int64_t va = 0,
                 int64_t vb = 0, int64_t vc = 0)
      : x(vx), y(vy), z(vz), a(va), b(vb), c(vc) {}

  bool operator==(const DESCRIPTOR_LOC &other) const {
    return (x == other.x && y == other.y && z == other.z && a == other.a &&
            b == other.b && c == other.c);
  }
};

// Hash value
namespace std {
template <> struct hash<VOXEL_LOC> {
  int64 operator()(const VOXEL_LOC &s) const {
    using std::hash;
    using std::size_t;
    return ((((s.z) * HASH_P) % MAX_N + (s.y)) * HASH_P) % MAX_N + (s.x);
  }
};
} // namespace std

// Hash value
namespace std {
template <> struct hash<DESCRIPTOR_LOC> {
  int64 operator()(const DESCRIPTOR_LOC &s) const {
    using std::hash;
    using std::size_t;
    // return ((((s.z) * HASH_P) % MAX_N + (s.y)) * HASH_P) % MAX_N + (s.x);
    return ((((((((((s.z) * HASH_P) % MAX_N + (s.y)) * HASH_P) % MAX_N +
                 (s.x)) *
                HASH_P) %
                   MAX_N +
               s.a) *
              HASH_P) %
                 MAX_N +
             s.b) *
            HASH_P) %
               MAX_N +
           s.c;
  }
};
} // namespace std

struct M_POINT {
  float xyz[3];
  float intensity;
  int count = 0;
};

double v3d_dis(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2) {
  Eigen::Vector3d p_inc = p1 - p2;
  return p_inc.norm();
  // return sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2) +
  //             pow(p1[2] - p2[2], 2));
}

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

double binary_similarity(const BinaryDescriptor &b1,
                         const BinaryDescriptor &b2) {
  double dis = 0;
  for (size_t i = 0; i < b1.occupy_array.size(); i++) {
    if (b1.occupy_array[i] == true && b2.occupy_array[i] == true) {
      dis += 1;
    }
  }
  return 2 * dis / (b1.summary + b2.summary);
}

void loadConfigSetting(std::string &config_file,
                       ConfigSetting &config_setting) {
  cv::FileStorage fSettings(config_file, cv::FileStorage::READ);
  if (!fSettings.isOpened()) {
    std::cerr << "Failed to open settings file at: " << config_file
              << std::endl;
    exit(-1);
  }
  config_setting.voxel_size = fSettings["VoxelSize"];
  config_setting.alternative_num = fSettings["alternative_num"];
  config_setting.sub_frame_num = fSettings["SubFrameNumber"];
  config_setting.planer_threshold = fSettings["PlanerThreshold"];
  config_setting.useful_corner_num = fSettings["useful_corner_num"];
  config_setting.icp_threshold = fSettings["icp_threshold"];
  config_setting.VoxelPointSize = fSettings["VoxelPointSize"];
  config_setting.ds_size = fSettings["ds_size"];
  config_setting.max_count_threshold = fSettings["max_count_threshold"];
  config_setting.descriptor_near_num = fSettings["descriptor_near_num"];
  config_setting.descriptor_min_len = fSettings["descriptor_min_len"];
  config_setting.descriptor_max_len = fSettings["descriptor_max_len"];
  config_setting.max_constraint_dis = fSettings["max_constrait_dis"];
  config_setting.triangle_resolution = fSettings["triangle_resolution"];
  std::cout << "Sucessfully load config file:" << config_file << std::endl;
}

// Similar with PCL voxelgrid filter
void down_sampling_voxel(pcl::PointCloud<pcl::PointXYZI> &pl_feat,
                         double voxel_size) {
  int intensity = rand() % 255;
  if (voxel_size < 0.01) {
    return;
  }
  std::unordered_map<VOXEL_LOC, M_POINT> feat_map;
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
    auto iter = feat_map.find(position);
    if (iter != feat_map.end()) {
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
      feat_map[position] = anp;
    }
  }
  plsize = feat_map.size();
  pl_feat.clear();
  pl_feat.resize(plsize);

  uint i = 0;
  for (auto iter = feat_map.begin(); iter != feat_map.end(); ++iter) {
    pl_feat[i].x = iter->second.xyz[0] / iter->second.count;
    pl_feat[i].y = iter->second.xyz[1] / iter->second.count;
    pl_feat[i].z = iter->second.xyz[2] / iter->second.count;
    pl_feat[i].intensity = iter->second.intensity / iter->second.count;
    i++;
  }
}
void down_sampling_voxel(pcl::PointCloud<pcl::PointXYZ> &pl_feat,
                         double voxel_size) {
  int intensity = rand() % 255;
  if (voxel_size < 0.01) {
    return;
  }
  std::unordered_map<VOXEL_LOC, M_POINT> feat_map;
  uint plsize = pl_feat.size();

  for (uint i = 0; i < plsize; i++) {
    pcl::PointXYZ &p_c = pl_feat[i];
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = p_c.data[j] / voxel_size;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }

    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                       (int64_t)loc_xyz[2]);
    auto iter = feat_map.find(position);
    if (iter != feat_map.end()) {
      iter->second.xyz[0] += p_c.x;
      iter->second.xyz[1] += p_c.y;
      iter->second.xyz[2] += p_c.z;
      iter->second.count++;
    } else {
      M_POINT anp;
      anp.xyz[0] = p_c.x;
      anp.xyz[1] = p_c.y;
      anp.xyz[2] = p_c.z;
      anp.count = 1;
      feat_map[position] = anp;
    }
  }
  plsize = feat_map.size();
  pl_feat.clear();
  pl_feat.resize(plsize);

  uint i = 0;
  for (auto iter = feat_map.begin(); iter != feat_map.end(); ++iter) {
    pl_feat[i].x = iter->second.xyz[0] / iter->second.count;
    pl_feat[i].y = iter->second.xyz[1] / iter->second.count;
    pl_feat[i].z = iter->second.xyz[2] / iter->second.count;
    i++;
  }
}
class OctoTree {
public:
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_ptr_;
  pcl::PointCloud<pcl::PointXYZI>::Ptr temp_cloud_;
  Plane *plane_ptr_;
  int layer_;
  int octo_state_; // 0 is end of tree, 1 is not
  int merge_num_ = 0;
  // x,y,z,-x,-y,-z
  bool is_project_ = false;
  std::vector<Eigen::Vector3d> project_normal;
  bool is_check_connect_[6];
  bool connect_[6];
  OctoTree *connect_tree_[6];
  bool is_publish_ = false;
  OctoTree *leaves_[8];
  double voxel_center_[3]; // x, y, z
  float quater_length_;
  float planer_threshold_;
  int points_size_threshold_;
  int update_size_threshold_;
  int new_points_;
  bool init_octo_;
  OctoTree(int points_size_threshold, float planer_threshold)
      : points_size_threshold_(points_size_threshold),
        planer_threshold_(planer_threshold) {
    cloud_ptr_ = pcl::PointCloud<pcl::PointXYZI>::Ptr(
        new pcl::PointCloud<pcl::PointXYZI>);
    temp_cloud_ = pcl::PointCloud<pcl::PointXYZI>::Ptr(
        new pcl::PointCloud<pcl::PointXYZI>);
    octo_state_ = 0;
    layer_ = 0;
    new_points_ = 0;
    update_size_threshold_ = 5;
    init_octo_ = false;
    for (int i = 0; i < 8; i++) {
      leaves_[i] = nullptr;
    }
    for (int i = 0; i < 6; i++) {
      is_check_connect_[i] = false;
      connect_[i] = false;
      connect_tree_[i] = nullptr;
    }
    plane_ptr_ = new Plane;
  }

  void init_planer(const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
                   Plane *plane) {
    plane->covariance = Eigen::Matrix3d::Zero();
    plane->center = Eigen::Vector3d::Zero();
    plane->normal = Eigen::Vector3d::Zero();
    plane->points_size = input_cloud->size();
    plane->radius = 0;
    // plane->cloud.clear();
    for (size_t i = 0; i < input_cloud->size(); i++) {
      Eigen::Vector3d pi(input_cloud->points[i].x, input_cloud->points[i].y,
                         input_cloud->points[i].z);
      plane->covariance += pi * pi.transpose();
      plane->center += pi;
      // plane->cloud.push_back(input_cloud->points[i]);
    }
    plane->center = plane->center / plane->points_size;
    plane->covariance = plane->covariance / plane->points_size -
                        plane->center * plane->center.transpose();
    Eigen::EigenSolver<Eigen::Matrix3d> es(plane->covariance);
    Eigen::Matrix3cd evecs = es.eigenvectors();
    Eigen::Vector3cd evals = es.eigenvalues();
    Eigen::Vector3d evalsReal; //注意这里定义的MatrixXd里没有c
    evalsReal = evals.real();  //获取特征值实数部分
    Eigen::Matrix3d::Index evalsMin, evalsMax;
    evalsReal.rowwise().sum().minCoeff(&evalsMin);
    evalsReal.rowwise().sum().maxCoeff(&evalsMax);
    int evalsMid = 3 - evalsMin - evalsMax;
    // std::cout << "min eigen value:" << evalsReal(evalsMin) << std::endl;
    //&& evalsReal(evalsMid) > 0.04
    double ratio = evalsReal(evalsMid) / evalsReal(evalsMin);
    if (evalsReal(evalsMin) < planer_threshold_) {
      plane->normal << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin),
          evecs.real()(2, evalsMin);
      plane->min_eigen_value = evalsReal(evalsMin);
      plane->radius = sqrt(evalsReal(evalsMax));
      plane->is_plane = true;

      // std::cout << "eigen value:" << evalsReal(0) << "," << evalsReal(1) <<
      // ","
      //           << evalsReal(2) << std::endl;
      plane->d = -(plane->normal(0) * plane->center(0) +
                   plane->normal(1) * plane->center(1) +
                   plane->normal(2) * plane->center(2));
      plane->p_center.x = plane->center(0);
      plane->p_center.y = plane->center(1);
      plane->p_center.z = plane->center(2);
      plane->p_center.normal_x = plane->normal(0);
      plane->p_center.normal_y = plane->normal(1);
      plane->p_center.normal_z = plane->normal(2);
      // for (size_t i = 0; i <)
      // if (fabs(plane->normal(2)) > 0.9) {
      //   plane->is_plane = true;
      //   if (plane->center(0) <= 13 && plane->center(0) >= 12 &&
      //       plane->center(1) <= -2 && plane->center(1) >= -3 &&
      //       plane->center(2) <= 1 && plane->center(2) >= -1) {
      //     plane->is_plane = true;
      //   } else {
      //     plane->is_plane = false;
      //   }
      // } else {
      //   plane->is_plane = false;
      // }
    } else {
      plane->is_plane = false;
    }
  }

  void init_octo_tree() {
    // down sample
    // down_sampling_voxel(*temp_cloud_, 0.05);
    if (temp_cloud_->points.size() > points_size_threshold_) {
      pcl::PointCloud<pcl::PointXYZI> test_cloud = *temp_cloud_;
      init_planer(test_cloud.makeShared(), plane_ptr_);
      if (plane_ptr_->is_plane == true) {
        octo_state_ = 0;
      } else {
        octo_state_ = 1;
        cut_octo_tree();
      }
      init_octo_ = true;
      new_points_ = 0;
      // temp_cloud_->clear();
    }
  }

  void cut_octo_tree() {
    if (layer_ >= max_layer) {
      octo_state_ = 0;
      return;
    }
    for (size_t i = 0; i < temp_cloud_->size(); i++) {
      int xyz[3] = {0, 0, 0};
      if (temp_cloud_->points[i].x > voxel_center_[0]) {
        xyz[0] = 1;
      }
      if (temp_cloud_->points[i].y > voxel_center_[1]) {
        xyz[1] = 1;
      }
      if (temp_cloud_->points[i].z > voxel_center_[2]) {
        xyz[2] = 1;
      }
      int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
      if (leaves_[leafnum] == nullptr) {
        leaves_[leafnum] =
            new OctoTree(points_size_threshold_ / 2, planer_threshold_);
        leaves_[leafnum]->voxel_center_[0] =
            voxel_center_[0] + (2 * xyz[0] - 1) * quater_length_;
        leaves_[leafnum]->voxel_center_[1] =
            voxel_center_[1] + (2 * xyz[1] - 1) * quater_length_;
        leaves_[leafnum]->voxel_center_[2] =
            voxel_center_[2] + (2 * xyz[2] - 1) * quater_length_;
        leaves_[leafnum]->quater_length_ = quater_length_ / 2;
        leaves_[leafnum]->layer_ = layer_ + 1;
      }
      leaves_[leafnum]->temp_cloud_->push_back(temp_cloud_->points[i]);
      leaves_[leafnum]->new_points_++;
    }
    for (uint i = 0; i < 8; i++) {
      if (leaves_[i] != nullptr) {
        if (leaves_[i]->temp_cloud_->size() >
            leaves_[i]->points_size_threshold_) {
          init_planer(leaves_[i]->temp_cloud_, leaves_[i]->plane_ptr_);
          if (leaves_[i]->plane_ptr_->is_plane) {
            leaves_[i]->octo_state_ = 0;
          } else {
            leaves_[i]->octo_state_ = 1;
            leaves_[i]->cut_octo_tree();
          }
          leaves_[i]->init_octo_ = true;
          leaves_[i]->new_points_ = 0;
          // leaves_[i]->temp_cloud_->clear();
        }
      }
    }
  }

  void UpdateOctoTree(const pcl::PointXYZI &pi) {
    if (!init_octo_) {
      new_points_++;
      temp_cloud_->push_back(pi);
      init_octo_tree();
    } else {
      if (plane_ptr_->is_plane) {
        new_points_++;
        temp_cloud_->push_back(pi);
        updatePlane();
      } else {
        if (layer_ < 2) {
          int xyz[3] = {0, 0, 0};
          if (pi.x > voxel_center_[0]) {
            xyz[0] = 1;
          }
          if (pi.y > voxel_center_[1]) {
            xyz[1] = 1;
          }
          if (pi.z > voxel_center_[2]) {
            xyz[2] = 1;
          }
          int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
          if (leaves_[leafnum] != nullptr) {
            leaves_[leafnum]->UpdateOctoTree(pi);
          } else {
            leaves_[leafnum] =
                new OctoTree(points_size_threshold_ / 2, planer_threshold_);
            leaves_[leafnum]->layer_ = layer_ + 1;
            leaves_[leafnum]->UpdateOctoTree(pi);
          }
        } else {
          if (temp_cloud_->size() > points_size_threshold_) {
            init_planer(temp_cloud_, plane_ptr_);
            temp_cloud_->clear();
          }
        }
      }
    }
  }

  void updatePlane() {
    if (temp_cloud_->size() >= update_size_threshold_) {
      Eigen::Matrix3d old_covariance = plane_ptr_->covariance;
      Eigen::Vector3d old_center = plane_ptr_->center;
      Eigen::Matrix3d sum_ppt =
          (plane_ptr_->covariance +
           plane_ptr_->center * plane_ptr_->center.transpose()) *
          plane_ptr_->points_size;
      Eigen::Vector3d sum_p = plane_ptr_->center * plane_ptr_->points_size;
      for (size_t i = 0; i < temp_cloud_->points.size(); i++) {
        Eigen::Vector3d pv(temp_cloud_->points[i].x, temp_cloud_->points[i].y,
                           temp_cloud_->points[i].z);
        sum_ppt += pv * pv.transpose();
        sum_p += pv;
      }
      plane_ptr_->points_size = plane_ptr_->points_size + temp_cloud_->size();
      plane_ptr_->center = sum_p / plane_ptr_->points_size;
      plane_ptr_->covariance =
          sum_ppt / plane_ptr_->points_size -
          plane_ptr_->center * plane_ptr_->center.transpose();
      Eigen::EigenSolver<Eigen::Matrix3d> es(plane_ptr_->covariance);
      Eigen::Matrix3cd evecs = es.eigenvectors();
      Eigen::Vector3cd evals = es.eigenvalues();
      Eigen::Vector3d evalsReal; //注意这里定义的MatrixXd里没有c
      evalsReal = evals.real();  //获取特征值实数部分
      Eigen::Matrix3d::Index evalsMin, evalsMax;
      evalsReal.rowwise().sum().minCoeff(&evalsMin);
      evalsReal.rowwise().sum().maxCoeff(&evalsMax);
      // std::cout << "min eigen value:" << evalsReal(evalsMin) <<
      // std::endl;
      if (evalsReal(evalsMin) < planer_threshold_) {
        plane_ptr_->normal << evecs.real()(0, evalsMin),
            evecs.real()(1, evalsMin), evecs.real()(2, evalsMin);
        plane_ptr_->min_eigen_value = evalsReal(evalsMin);
        plane_ptr_->radius = sqrt(evalsReal(evalsMax));
        plane_ptr_->d = -(plane_ptr_->normal(0) * plane_ptr_->center(0) +
                          plane_ptr_->normal(1) * plane_ptr_->center(1) +
                          plane_ptr_->normal(2) * plane_ptr_->center(2));
        plane_ptr_->p_center.x = plane_ptr_->center(0);
        plane_ptr_->p_center.y = plane_ptr_->center(1);
        plane_ptr_->p_center.z = plane_ptr_->center(2);
        plane_ptr_->p_center.normal_x = plane_ptr_->normal(0);
        plane_ptr_->p_center.normal_y = plane_ptr_->normal(1);
        plane_ptr_->p_center.normal_z = plane_ptr_->normal(2);
        plane_ptr_->is_plane = true;
        temp_cloud_->clear();
        new_points_ = 0;
      } else {
        // plane->is_plane = false;
        plane_ptr_->covariance = old_covariance;
        plane_ptr_->center = old_center;
        plane_ptr_->points_size = plane_ptr_->points_size - temp_cloud_->size();
        temp_cloud_->clear();
        new_points_ = 0;
      }
    }
  }
};

void buildUnorderMap(const pcl::PointCloud<pcl::PointXYZI> &pl_feat,
                     const float voxel_size, const int points_size_threshold,
                     const float planer_threshold,
                     std::unordered_map<VOXEL_LOC, OctoTree *> &feat_map) {
  uint plsize = pl_feat.size();
  for (uint i = 0; i < plsize; i++) {
    const pcl::PointXYZI &p_c = pl_feat[i];
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = p_c.data[j] / voxel_size;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }
    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                       (int64_t)loc_xyz[2]);
    auto iter = feat_map.find(position);
    if (iter != feat_map.end()) {
      feat_map[position]->temp_cloud_->points.push_back(p_c);
    } else {
      OctoTree *octo_tree =
          new OctoTree(points_size_threshold, planer_threshold);
      feat_map[position] = octo_tree;
      feat_map[position]->temp_cloud_->points.push_back(p_c);
    }
  }
  std::vector<std::unordered_map<VOXEL_LOC, OctoTree *>::iterator> iter_list;
  std::vector<size_t> index;
  size_t i = 0;
  for (auto iter = feat_map.begin(); iter != feat_map.end(); ++iter) {
    index.push_back(i);
    i++;
    iter_list.push_back(iter);
    // iter->second->init_octo_tree();
  }
  std::for_each(
      std::execution::par_unseq, index.begin(), index.end(),
      [&](const size_t &i) { iter_list[i]->second->init_octo_tree(); });
}

void updateUnorderMap(const pcl::PointCloud<pcl::PointXYZI> &pl_feat,
                      const float voxel_size, const int points_size_threshold,
                      const float planer_threshold,
                      std::unordered_map<VOXEL_LOC, OctoTree *> &feat_map) {
  uint plsize = pl_feat.size();
  for (uint i = 0; i < plsize; i++) {
    const pcl::PointXYZI &p_c = pl_feat[i];
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = p_c.data[j] / voxel_size;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }
    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                       (int64_t)loc_xyz[2]);
    auto iter = feat_map.find(position);
    if (iter != feat_map.end()) {
      feat_map[position]->UpdateOctoTree(p_c);
    } else {
      OctoTree *octo_tree =
          new OctoTree(points_size_threshold, planer_threshold);
      feat_map[position] = octo_tree;
      feat_map[position]->quater_length_ = voxel_size;
      feat_map[position]->voxel_center_[0] = (0.5 + position.x) * voxel_size;
      feat_map[position]->voxel_center_[0] = (0.5 + position.y) * voxel_size;
      feat_map[position]->voxel_center_[0] = (0.5 + position.z) * voxel_size;
      feat_map[position]->UpdateOctoTree(p_c);
    }
  }
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

void NormalToQuaternion(const Eigen::Vector3d &normal_vec,
                        geometry_msgs::Quaternion &q) {
  float CosY = normal_vec(2) / sqrt(normal_vec(0) * normal_vec(0) +
                                    normal_vec(1) * normal_vec(1));
  float CosYDiv2 = sqrt((CosY + 1) / 2);
  if (normal_vec(0) < 0)
    CosYDiv2 = -CosYDiv2;
  float SinYDiv2 = sqrt((1 - CosY) / 2);
  float CosX =
      sqrt((normal_vec(0) * normal_vec(0) + normal_vec(2) * normal_vec(2)) /
           (normal_vec(0) * normal_vec(0) + normal_vec(1) * normal_vec(1) +
            normal_vec(2) * normal_vec(2)));
  if (normal_vec(2) < 0)
    CosX = -CosX;
  float CosXDiv2 = sqrt((CosX + 1) / 2);
  if (normal_vec(1) < 0)
    CosXDiv2 = -CosXDiv2;
  float SinXDiv2 = sqrt((1 - CosX) / 2);
  q.w = CosXDiv2 * CosYDiv2;
  q.x = SinXDiv2 * CosYDiv2;
  q.y = CosXDiv2 * SinYDiv2;
  q.z = -SinXDiv2 * SinYDiv2;
}

// struct TriangleMatchSolver {
//   TriangleMatchSolver(Eigen::Vector3d p1_, Eigen::Vector3d p2_,
//                       Eigen::Vector3d p3_, Eigen::Vector3d p11_,
//                       Eigen::Vector3d p22_, Eigen::Vector3d p33_);
//   Eigen::Vector3d p1;
//   Eigen::Vector3d p2;
//   Eigen::Vector3d p3;
//   Eigen::Vector3d p11;
//   Eigen::Vector3d p22;
//   Eigen::Vector3d p33;
// }

struct IcpSolver {
  IcpSolver(Eigen::Vector3d curr_point_, Eigen::Vector3d plane_unit_norm_)
      : curr_point(curr_point_), plane_unit_norm(plane_unit_norm_){};

  template <typename T>
  bool operator()(const T *q, const T *t, T *residual) const {
    Eigen::Quaternion<T> q_w_curr{q[3], q[0], q[1], q[2]};
    Eigen::Matrix<T, 3, 1> t_w_curr{t[0], t[1], t[2]};
    Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()),
                              T(curr_point.z())};
    Eigen::Matrix<T, 3, 1> point_w;
    point_w = q_w_curr * cp + t_w_curr;
    Eigen::Matrix<T, 3, 1> norm(T(plane_unit_norm.x()), T(plane_unit_norm.y()),
                                T(plane_unit_norm.z()));
    residual[0] = point_w[0] - norm[0];
    residual[1] = point_w[1] - norm[1];
    residual[2] = point_w[2] - norm[2];
    return true;
  }
  static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_,
                                     const Eigen::Vector3d plane_unit_norm_) {
    return (new ceres::AutoDiffCostFunction<IcpSolver, 3, 4, 3>(
        new IcpSolver(curr_point_, plane_unit_norm_)));
  }

  Eigen::Vector3d curr_point;
  Eigen::Vector3d plane_unit_norm;
};

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

struct LidarPlaneSolver {
  LidarPlaneSolver(Eigen::Vector3d curr_point_,
                   Eigen::Vector3d plane_unit_norm_,
                   double negative_OA_dot_norm_)
      : curr_point(curr_point_), plane_unit_norm(plane_unit_norm_),
        negative_OA_dot_norm(negative_OA_dot_norm_){};

  template <typename T>
  bool operator()(const T *q, const T *t, T *residual) const {
    Eigen::Quaternion<T> q_w_curr{q[3], q[0], q[1], q[2]};
    Eigen::Matrix<T, 3, 1> t_w_curr{t[0], t[1], t[2]};
    Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()),
                              T(curr_point.z())};
    Eigen::Matrix<T, 3, 1> point_w;
    point_w = q_w_curr * cp + t_w_curr;
    Eigen::Matrix<T, 3, 1> norm(T(plane_unit_norm.x()), T(plane_unit_norm.y()),
                                T(plane_unit_norm.z()));
    residual[0] = norm.dot(point_w) + T(negative_OA_dot_norm);
    return true;
  }
  static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_,
                                     const Eigen::Vector3d plane_unit_norm_,
                                     const double negative_OA_dot_norm_) {
    return (new ceres::AutoDiffCostFunction<LidarPlaneSolver, 1, 4, 3>(
        new LidarPlaneSolver(curr_point_, plane_unit_norm_,
                             negative_OA_dot_norm_)));
  }

  Eigen::Vector4d plane_params;
  Eigen::Vector3d curr_point;
  Eigen::Vector3d plane_unit_norm;
  double negative_OA_dot_norm;
};

void rgb2grey(const cv::Mat &rgb_image, cv::Mat &grey_img) {
  for (int x = 0; x < rgb_image.cols; x++) {
    for (int y = 0; y < rgb_image.rows; y++) {
      grey_img.at<uchar>(y, x) = 1.0 / 3.0 * rgb_image.at<cv::Vec3b>(y, x)[0] +
                                 1.0 / 3.0 * rgb_image.at<cv::Vec3b>(y, x)[1] +
                                 1.0 / 3.0 * rgb_image.at<cv::Vec3b>(y, x)[2];
    }
  }
}

void mapJet(double v, double vmin, double vmax, uint8_t &r, uint8_t &g,
            uint8_t &b) {
  r = 255;
  g = 255;
  b = 255;

  if (v < vmin) {
    v = vmin;
  }

  if (v > vmax) {
    v = vmax;
  }

  double dr, dg, db;

  if (v < 0.1242) {
    db = 0.504 + ((1. - 0.504) / 0.1242) * v;
    dg = dr = 0.;
  } else if (v < 0.3747) {
    db = 1.;
    dr = 0.;
    dg = (v - 0.1242) * (1. / (0.3747 - 0.1242));
  } else if (v < 0.6253) {
    db = (0.6253 - v) * (1. / (0.6253 - 0.3747));
    dg = 1.;
    dr = (v - 0.3747) * (1. / (0.6253 - 0.3747));
  } else if (v < 0.8758) {
    db = 0.;
    dr = 1.;
    dg = (0.8758 - v) * (1. / (0.8758 - 0.6253));
  } else {
    db = 0.;
    dg = 0.;
    dr = 1. - (v - 0.8758) * ((1. - 0.504) / (1. - 0.8758));
  }

  r = (uint8_t)(255 * dr);
  g = (uint8_t)(255 * dg);
  b = (uint8_t)(255 * db);
}

void pubNormal(const ros::Publisher &normal_pub, const std::string normal_ns,
               const int normal_id, const pcl::PointXYZINormal normal_p,
               const Eigen::Vector3d rgb) {
  visualization_msgs::Marker normal;
  normal.header.frame_id = "camera_init";
  normal.header.stamp = ros::Time();
  normal.ns = normal_ns;
  normal.id = normal_id;
  normal.type = visualization_msgs::Marker::ARROW;
  normal.action = visualization_msgs::Marker::ADD;
  normal.pose.position.x = normal_p.x;
  normal.pose.position.y = normal_p.y;
  normal.pose.position.z = normal_p.z;
  geometry_msgs::Quaternion q;
  Eigen::Vector3d normal_vec(normal_p.normal_x, normal_p.normal_y,
                             normal_p.normal_z);
  CalcQuation(normal_vec, 0, q);
  normal.pose.orientation = q;
  normal.scale.x = 1;
  normal.scale.y = 0.2;
  normal.scale.z = 0.2;
  normal.color.a = 0.8; // Don't forget to set the alpha!
  normal.color.r = rgb(0);
  normal.color.g = rgb(1);
  normal.color.b = rgb(2);
  normal.lifetime = ros::Duration();
  normal_pub.publish(normal);
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

void pubPlaneMap(const std::unordered_map<VOXEL_LOC, OctoTree *> &feat_map,
                 const ros::Publisher &plane_map_pub) {
  int normal_id = 0;
  int plane_count = 0;
  int sub_plane_count = 0;
  int sub_sub_plane_count = 0;
  OctoTree *current_octo = nullptr;
  float plane_threshold = 0.0025;
  ros::Rate loop(50);
  for (auto iter = feat_map.begin(); iter != feat_map.end(); iter++) {
    if (iter->second->octo_state_ == 0 && iter->second->plane_ptr_->is_plane) {
      Eigen::Vector3d normal_rgb(0.0, 1.0, 0.0);
      uint8_t r, g, b;
      mapJet(iter->second->plane_ptr_->min_eigen_value / plane_threshold, 0, 1,
             r, g, b);
      Eigen::Vector3d plane_rgb(r / 256.0, g / 256.0, b / 256.0);
      pubNormal(plane_map_pub, "normal", normal_id,
                iter->second->plane_ptr_->p_center, normal_rgb);
      pubPlane(plane_map_pub, "plane", normal_id,
               iter->second->plane_ptr_->p_center,
               iter->second->plane_ptr_->radius, plane_rgb);
      normal_id++;
      plane_count++;
      loop.sleep();
    } else {
      for (uint i = 0; i < 8; i++) {
        if (iter->second->leaves_[i] != nullptr) {
          if (iter->second->leaves_[i]->octo_state_ == 0 &&
              iter->second->leaves_[i]->plane_ptr_->is_plane) {
            Eigen::Vector3d normal_rgb(0.0, 1.0, 0.0);
            uint8_t r, g, b;
            mapJet(iter->second->leaves_[i]->plane_ptr_->min_eigen_value /
                       plane_threshold,
                   0, 1, r, g, b);
            Eigen::Vector3d plane_rgb(r / 256.0, g / 256.0, b / 256.0);
            pubNormal(plane_map_pub, "normal", normal_id,
                      iter->second->leaves_[i]->plane_ptr_->p_center,
                      normal_rgb);
            pubPlane(plane_map_pub, "plane", normal_id,
                     iter->second->leaves_[i]->plane_ptr_->p_center,
                     iter->second->leaves_[i]->plane_ptr_->radius, plane_rgb);
            sub_plane_count++;
            normal_id++;
            loop.sleep();
          } else {
            OctoTree *temp_octo_tree = iter->second->leaves_[i];
            for (uint j = 0; j < 8; j++) {
              if (temp_octo_tree->leaves_[j] != nullptr) {
                if (temp_octo_tree->leaves_[j]->octo_state_ == 0 &&
                    temp_octo_tree->leaves_[j]->plane_ptr_->is_plane) {
                  Eigen::Vector3d normal_rgb(0.0, 1.0, 0.0);
                  uint8_t r, g, b;
                  mapJet(
                      temp_octo_tree->leaves_[i]->plane_ptr_->min_eigen_value /
                          plane_threshold,
                      0, 1, r, g, b);
                  Eigen::Vector3d plane_rgb(r / 256.0, g / 256.0, b / 256.0);
                  pubNormal(plane_map_pub, "normal", normal_id,
                            temp_octo_tree->leaves_[i]->plane_ptr_->p_center,
                            normal_rgb);
                  pubPlane(plane_map_pub, "plane", normal_id,
                           temp_octo_tree->leaves_[i]->plane_ptr_->p_center,
                           temp_octo_tree->leaves_[i]->plane_ptr_->radius,
                           plane_rgb);
                  sub_sub_plane_count++;
                  normal_id++;
                  loop.sleep();
                }
              }
            }
          }
        }
      }
    }
  }
  std::cout << "Plane counts:" << plane_count << std::endl;
  std::cout << "Sub Plane counts:" << sub_plane_count << std::endl;
  std::cout << "Sub Sub Plane counts:" << sub_sub_plane_count << std::endl;
}

void getProjectionPlane(std::unordered_map<VOXEL_LOC, OctoTree *> &feat_map,
                        const double dis_threshold,
                        const double normal_threshold,
                        std::vector<Plane *> &project_plane_list) {
  std::vector<Plane *> origin_list;
  for (auto iter = feat_map.begin(); iter != feat_map.end(); iter++) {
    if (iter->second->plane_ptr_->is_plane) {
      origin_list.push_back(iter->second->plane_ptr_);
    }
  }
  for (size_t i = 0; i < origin_list.size(); i++)
    origin_list[i]->id = 0;
  int current_id = 1;
  for (auto iter = origin_list.end() - 1; iter != origin_list.begin(); iter--) {
    for (auto iter2 = origin_list.begin(); iter2 != iter; iter2++) {
      Eigen::Vector3d normal_diff = (*iter)->normal - (*iter2)->normal;
      Eigen::Vector3d normal_add = (*iter)->normal + (*iter2)->normal;
      double dis1 = fabs((*iter)->normal(0) * (*iter2)->center(0) +
                         (*iter)->normal(1) * (*iter2)->center(1) +
                         (*iter)->normal(2) * (*iter2)->center(2) + (*iter)->d);
      double dis2 =
          fabs((*iter2)->normal(0) * (*iter)->center(0) +
               (*iter2)->normal(1) * (*iter)->center(1) +
               (*iter2)->normal(2) * (*iter)->center(2) + (*iter2)->d);
      if (normal_diff.norm() < normal_threshold ||
          normal_add.norm() < normal_threshold)
        if (dis1 < dis_threshold && dis2 < dis_threshold) {
          if ((*iter)->id == 0 && (*iter2)->id == 0) {
            (*iter)->id = current_id;
            (*iter2)->id = current_id;
            current_id++;
          } else if ((*iter)->id == 0 && (*iter2)->id != 0)
            (*iter)->id = (*iter2)->id;
          else if ((*iter)->id != 0 && (*iter2)->id == 0)
            (*iter2)->id = (*iter)->id;
        }
    }
  }
  std::vector<Plane *> merge_list;
  std::vector<int> merge_flag;

  for (size_t i = 0; i < origin_list.size(); i++) {
    auto it =
        std::find(merge_flag.begin(), merge_flag.end(), origin_list[i]->id);
    if (it != merge_flag.end())
      continue;
    if (origin_list[i]->id == 0) {
      continue;
    }
    Plane *merge_plane = new Plane;
    (*merge_plane) = (*origin_list[i]);
    bool is_merge = false;
    for (size_t j = 0; j < origin_list.size(); j++) {
      if (i == j)
        continue;
      if (origin_list[j]->id == origin_list[i]->id) {
        is_merge = true;
        Eigen::Matrix3d P_PT1 =
            (merge_plane->covariance +
             merge_plane->center * merge_plane->center.transpose()) *
            merge_plane->points_size;
        Eigen::Matrix3d P_PT2 =
            (origin_list[j]->covariance +
             origin_list[j]->center * origin_list[j]->center.transpose()) *
            origin_list[j]->points_size;
        Eigen::Vector3d merge_center =
            (merge_plane->center * merge_plane->points_size +
             origin_list[j]->center * origin_list[j]->points_size) /
            (merge_plane->points_size + origin_list[j]->points_size);
        Eigen::Matrix3d merge_covariance =
            (P_PT1 + P_PT2) /
                (merge_plane->points_size + origin_list[j]->points_size) -
            merge_center * merge_center.transpose();
        merge_plane->covariance = merge_covariance;
        merge_plane->center = merge_center;
        merge_plane->points_size =
            merge_plane->points_size + origin_list[j]->points_size;
        merge_plane->sub_plane_num++;
        // for (size_t k = 0; k < origin_list[j]->cloud.size(); k++) {
        //   merge_plane->cloud.points.push_back(origin_list[j]->cloud.points[k]);
        // }
        Eigen::EigenSolver<Eigen::Matrix3d> es(merge_plane->covariance);
        Eigen::Matrix3cd evecs = es.eigenvectors();
        Eigen::Vector3cd evals = es.eigenvalues();
        Eigen::Vector3d evalsReal;
        evalsReal = evals.real();
        Eigen::Matrix3f::Index evalsMin, evalsMax;
        evalsReal.rowwise().sum().minCoeff(&evalsMin);
        evalsReal.rowwise().sum().maxCoeff(&evalsMax);
        Eigen::Vector3d evecMin = evecs.real().col(evalsMin);
        merge_plane->normal << evecs.real()(0, evalsMin),
            evecs.real()(1, evalsMin), evecs.real()(2, evalsMin);
        merge_plane->radius = sqrt(evalsReal(evalsMax));
        merge_plane->d = -(merge_plane->normal(0) * merge_plane->center(0) +
                           merge_plane->normal(1) * merge_plane->center(1) +
                           merge_plane->normal(2) * merge_plane->center(2));
        merge_plane->p_center.x = merge_plane->center(0);
        merge_plane->p_center.y = merge_plane->center(1);
        merge_plane->p_center.z = merge_plane->center(2);
        merge_plane->p_center.normal_x = merge_plane->normal(0);
        merge_plane->p_center.normal_y = merge_plane->normal(1);
        merge_plane->p_center.normal_z = merge_plane->normal(2);
      }
    }
    if (is_merge) {
      merge_flag.push_back(merge_plane->id);
      merge_list.push_back(merge_plane);
    }
  }
  project_plane_list = merge_list;
}

void mergePlane(const double dis_threshold, const double normal_threshold,
                std::vector<Plane *> &origin_list,
                std::vector<Plane *> &merge_plane_list) {
  for (size_t i = 0; i < origin_list.size(); i++)
    origin_list[i]->id = 0;
  int current_id = 1;
  for (auto iter = origin_list.end() - 1; iter != origin_list.begin(); iter--) {
    for (auto iter2 = origin_list.begin(); iter2 != iter; iter2++) {
      Eigen::Vector3d normal_diff = (*iter)->normal - (*iter2)->normal;
      Eigen::Vector3d normal_add = (*iter)->normal + (*iter2)->normal;
      double dis1 = fabs((*iter)->normal(0) * (*iter2)->center(0) +
                         (*iter)->normal(1) * (*iter2)->center(1) +
                         (*iter)->normal(2) * (*iter2)->center(2) + (*iter)->d);
      double dis2 =
          fabs((*iter2)->normal(0) * (*iter)->center(0) +
               (*iter2)->normal(1) * (*iter)->center(1) +
               (*iter2)->normal(2) * (*iter)->center(2) + (*iter2)->d);
      if (normal_diff.norm() < normal_threshold ||
          normal_add.norm() < normal_threshold)
        if (dis1 < dis_threshold && dis2 < dis_threshold) {
          if ((*iter)->id == 0 && (*iter2)->id == 0) {
            (*iter)->id = current_id;
            (*iter2)->id = current_id;
            current_id++;
          } else if ((*iter)->id == 0 && (*iter2)->id != 0)
            (*iter)->id = (*iter2)->id;
          else if ((*iter)->id != 0 && (*iter2)->id == 0)
            (*iter2)->id = (*iter)->id;
        }
    }
  }
  std::vector<int> merge_flag;

  for (size_t i = 0; i < origin_list.size(); i++) {
    auto it =
        std::find(merge_flag.begin(), merge_flag.end(), origin_list[i]->id);
    if (it != merge_flag.end())
      continue;
    if (origin_list[i]->id == 0) {
      merge_plane_list.push_back(origin_list[i]);
      continue;
    }
    Plane *merge_plane = new Plane;
    (*merge_plane) = (*origin_list[i]);
    bool is_merge = false;
    for (size_t j = 0; j < origin_list.size(); j++) {
      if (i == j)
        continue;
      if (origin_list[j]->id == origin_list[i]->id) {
        is_merge = true;
        Eigen::Matrix3d P_PT1 =
            (merge_plane->covariance +
             merge_plane->center * merge_plane->center.transpose()) *
            merge_plane->points_size;
        Eigen::Matrix3d P_PT2 =
            (origin_list[j]->covariance +
             origin_list[j]->center * origin_list[j]->center.transpose()) *
            origin_list[j]->points_size;
        Eigen::Vector3d merge_center =
            (merge_plane->center * merge_plane->points_size +
             origin_list[j]->center * origin_list[j]->points_size) /
            (merge_plane->points_size + origin_list[j]->points_size);
        Eigen::Matrix3d merge_covariance =
            (P_PT1 + P_PT2) /
                (merge_plane->points_size + origin_list[j]->points_size) -
            merge_center * merge_center.transpose();
        merge_plane->covariance = merge_covariance;
        merge_plane->center = merge_center;
        merge_plane->points_size =
            merge_plane->points_size + origin_list[j]->points_size;
        merge_plane->sub_plane_num += origin_list[j]->sub_plane_num;
        // for (size_t k = 0; k < origin_list[j]->cloud.size(); k++) {
        //   merge_plane->cloud.points.push_back(origin_list[j]->cloud.points[k]);
        // }
        Eigen::EigenSolver<Eigen::Matrix3d> es(merge_plane->covariance);
        Eigen::Matrix3cd evecs = es.eigenvectors();
        Eigen::Vector3cd evals = es.eigenvalues();
        Eigen::Vector3d evalsReal;
        evalsReal = evals.real();
        Eigen::Matrix3f::Index evalsMin, evalsMax;
        evalsReal.rowwise().sum().minCoeff(&evalsMin);
        evalsReal.rowwise().sum().maxCoeff(&evalsMax);
        Eigen::Vector3d evecMin = evecs.real().col(evalsMin);
        merge_plane->normal << evecs.real()(0, evalsMin),
            evecs.real()(1, evalsMin), evecs.real()(2, evalsMin);
        merge_plane->radius = sqrt(evalsReal(evalsMax));
        merge_plane->d = -(merge_plane->normal(0) * merge_plane->center(0) +
                           merge_plane->normal(1) * merge_plane->center(1) +
                           merge_plane->normal(2) * merge_plane->center(2));
        merge_plane->p_center.x = merge_plane->center(0);
        merge_plane->p_center.y = merge_plane->center(1);
        merge_plane->p_center.z = merge_plane->center(2);
        merge_plane->p_center.normal_x = merge_plane->normal(0);
        merge_plane->p_center.normal_y = merge_plane->normal(1);
        merge_plane->p_center.normal_z = merge_plane->normal(2);
      }
    }
    if (is_merge) {
      merge_flag.push_back(merge_plane->id);
      merge_plane_list.push_back(merge_plane);
    }
  }
  // if (merge_plane_list.size() == 0) {
  //   for (auto var : origin_list) {
  //     merge_plane_list.push_back(var);
  //   }
  // }
}

void BuildConnection(const float voxel_size, const float normal_threshold,
                     std::unordered_map<VOXEL_LOC, OctoTree *> &feat_map) {
  for (auto iter = feat_map.begin(); iter != feat_map.end(); iter++) {
    if (iter->second->plane_ptr_->is_plane) {
      OctoTree *current_octo = iter->second;
      for (int i = 0; i < 6; i++) {
        VOXEL_LOC neighbor = iter->first;
        if (i == 0) {
          neighbor.x = neighbor.x + 1;
        } else if (i == 1) {
          neighbor.y = neighbor.y + 1;
        } else if (i == 2) {
          neighbor.z = neighbor.z + 1;
        } else if (i == 3) {
          neighbor.x = neighbor.x - 1;
        } else if (i == 4) {
          neighbor.y = neighbor.y - 1;
        } else if (i == 5) {
          neighbor.z = neighbor.z - 1;
        }
        auto near = feat_map.find(neighbor);
        if (near == feat_map.end()) {
          current_octo->is_check_connect_[i] = true;
          current_octo->connect_[i] = false;
        } else {
          if (!current_octo->is_check_connect_[i]) {
            OctoTree *near_octo = near->second;
            current_octo->is_check_connect_[i] = true;
            int j;
            if (i >= 3) {
              j = i - 3;
            } else {
              j = i + 3;
            }
            near_octo->is_check_connect_[j] = true;
            if (near_octo->plane_ptr_->is_plane) {
              // merge near octo
              Eigen::Vector3d normal_diff = current_octo->plane_ptr_->normal -
                                            near_octo->plane_ptr_->normal;
              Eigen::Vector3d normal_add = current_octo->plane_ptr_->normal +
                                           near_octo->plane_ptr_->normal;
              if (normal_diff.norm() < normal_threshold ||
                  normal_add.norm() < normal_threshold) {
                current_octo->connect_[i] = true;
                near_octo->connect_[j] = true;
                current_octo->connect_tree_[i] = near_octo;
                near_octo->connect_tree_[j] = current_octo;
              } else {
                current_octo->connect_[i] = false;
                near_octo->connect_[j] = false;
              }
            } else {
              current_octo->connect_[i] = false;
              near_octo->connect_[j] = true;
              near_octo->connect_tree_[j] = current_octo;
            }
          }
        }
      }
    }
  }
}

void build_triangle_descriptor(const std::vector<BinaryDescriptor> &binary_list,
                               const int frame_number, const int near_num,
                               const double min_dis_threshold,
                               const double max_dis_threshold,
                               const double grid_size,
                               std::vector<Descriptor> &descriptor_list) {
  float scale = 1 / grid_size;
  std::unordered_map<VOXEL_LOC, bool> feat_map;
  pcl::PointCloud<pcl::PointXYZ> key_cloud;
  for (auto var : binary_list) {
    pcl::PointXYZ pi;
    pi.x = var.location[0];
    pi.y = var.location[1];
    pi.z = var.location[2];
    key_cloud.push_back(pi);
  }
  pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kd_tree(
      new pcl::KdTreeFLANN<pcl::PointXYZ>);
  kd_tree->setInputCloud(key_cloud.makeShared());
  int K = near_num;
  // 创建两个向量，分别存放近邻的索引值、近邻的中心距
  std::vector<int> pointIdxNKNSearch(near_num);
  std::vector<float> pointNKNSquaredDistance(near_num);
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
          if (a > max_dis_threshold || b > max_dis_threshold ||
              c > max_dis_threshold || a < min_dis_threshold ||
              b < min_dis_threshold || c < min_dis_threshold) {
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
            Descriptor single_descriptor;
            single_descriptor.A = A;
            single_descriptor.B = B;
            single_descriptor.C = C;
            single_descriptor.count_A = binary_A.summary;
            single_descriptor.count_B = binary_B.summary;
            single_descriptor.count_C = binary_C.summary;
            single_descriptor.binary_A = binary_A;
            single_descriptor.binary_B = binary_B;
            single_descriptor.binary_C = binary_C;
            single_descriptor.center = (A + B + C) / 3;
            single_descriptor.triangle << scale * a, scale * b, scale * c;
            // single_descriptor.angle[0] = fabs(5 * normal_1.dot(normal_2));
            // single_descriptor.angle[1] = fabs(5 * normal_1.dot(normal_3));
            // single_descriptor.angle[2] = fabs(5 * normal_3.dot(normal_2));
            single_descriptor.angle << 0, 0, 0;
            single_descriptor.frame_number = frame_number;
            single_descriptor.score_frame.push_back(frame_number);
            Eigen::Matrix3d triangle_positon;
            triangle_positon.block<3, 1>(0, 0) = A;
            triangle_positon.block<3, 1>(0, 1) = B;
            triangle_positon.block<3, 1>(0, 2) = C;
            single_descriptor.position_list.push_back(triangle_positon);
            single_descriptor.triangle_scale = scale;
            feat_map[position] = true;
            descriptor_list.push_back(single_descriptor);
          }
        }
      }
    }
  }
}

void build_descriptor(
    const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &key_cloud,
    std::vector<double> &count_list, const int frame_number, const int near_num,
    const double min_dis_threshold, const double max_dis_threshold,
    const double grid_size, std::vector<Descriptor> &descriptor_list) {
  double normal_threshold = 0.5;
  float scale = 1 / grid_size;
  std::unordered_map<VOXEL_LOC, bool> feat_map;
  pcl::KdTreeFLANN<pcl::PointXYZINormal>::Ptr kd_tree(
      new pcl::KdTreeFLANN<pcl::PointXYZINormal>);
  kd_tree->setInputCloud(key_cloud);
  int K = near_num;
  double count_A = 0;
  double count_B = 0;
  double count_C = 0;
  // 创建两个向量，分别存放近邻的索引值、近邻的中心距
  std::vector<int> pointIdxNKNSearch(near_num);
  std::vector<float> pointNKNSquaredDistance(near_num);
  for (size_t i = 0; i < key_cloud->size(); i++) {
    pcl::PointXYZINormal searchPoint = key_cloud->points[i];
    if (kd_tree->nearestKSearch(searchPoint, K, pointIdxNKNSearch,
                                pointNKNSquaredDistance) > 0) {
      for (int m = 1; m < K - 1; m++) {
        for (int n = m + 1; n < K; n++) {
          pcl::PointXYZINormal p1 = searchPoint;
          pcl::PointXYZINormal p2 = key_cloud->points[pointIdxNKNSearch[m]];
          pcl::PointXYZINormal p3 = key_cloud->points[pointIdxNKNSearch[n]];
          Eigen::Vector3d normal_inc1(p1.normal_x - p2.normal_x,
                                      p1.normal_y - p2.normal_y,
                                      p1.normal_z - p2.normal_z);
          Eigen::Vector3d normal_inc2(p3.normal_x - p2.normal_x,
                                      p3.normal_y - p2.normal_y,
                                      p3.normal_z - p2.normal_z);
          Eigen::Vector3d normal_add1(p1.normal_x + p2.normal_x,
                                      p1.normal_y + p2.normal_y,
                                      p1.normal_z + p2.normal_z);
          Eigen::Vector3d normal_add2(p3.normal_x + p2.normal_x,
                                      p3.normal_y + p2.normal_y,
                                      p3.normal_z + p2.normal_z);
          // if (normal_inc1.norm() > normal_threshold ||
          //     normal_add1.norm() < normal_threshold) {
          //   continue;
          // }
          // if (normal_inc2.norm() > normal_threshold ||
          //     normal_add2.norm() < normal_threshold) {
          //   continue;
          // }
          double a = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) +
                          pow(p1.z - p2.z, 2));
          double b = sqrt(pow(p1.x - p3.x, 2) + pow(p1.y - p3.y, 2) +
                          pow(p1.z - p3.z, 2));
          double c = sqrt(pow(p3.x - p2.x, 2) + pow(p3.y - p2.y, 2) +
                          pow(p3.z - p2.z, 2));
          if (a > max_dis_threshold || b > max_dis_threshold ||
              c > max_dis_threshold || a < min_dis_threshold ||
              b < min_dis_threshold || c < min_dis_threshold) {
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
          pcl::PointXYZ d_p;
          d_p.x = a * 1000;
          d_p.y = b * 1000;
          d_p.z = c * 1000;
          VOXEL_LOC position((int64_t)d_p.x, (int64_t)d_p.y, (int64_t)d_p.z);
          auto iter = feat_map.find(position);
          Eigen::Vector3d normal_1, normal_2, normal_3;
          if (iter == feat_map.end()) {
            if (l1[0] == l2[0]) {
              A << p1.x, p1.y, p1.z;
              count_A = count_list[i];
              normal_1 << p1.normal_x, p1.normal_y, p1.normal_z;
            } else if (l1[1] == l2[1]) {
              A << p2.x, p2.y, p2.z;
              count_A = count_list[pointIdxNKNSearch[m]];
              normal_1 << p2.normal_x, p2.normal_y, p2.normal_z;
            } else {
              A << p3.x, p3.y, p3.z;
              count_A = count_list[pointIdxNKNSearch[n]];
              normal_1 << p3.normal_x, p3.normal_y, p3.normal_z;
            }
            if (l1[0] == l3[0]) {
              B << p1.x, p1.y, p1.z;
              count_B = count_list[i];
              normal_2 << p1.normal_x, p1.normal_y, p1.normal_z;
            } else if (l1[1] == l3[1]) {
              B << p2.x, p2.y, p2.z;
              count_B = count_list[pointIdxNKNSearch[m]];
              normal_2 << p2.normal_x, p2.normal_y, p2.normal_z;
            } else {
              B << p3.x, p3.y, p3.z;
              count_B = count_list[pointIdxNKNSearch[n]];
              normal_2 << p3.normal_x, p3.normal_y, p3.normal_z;
            }
            if (l2[0] == l3[0]) {
              C << p1.x, p1.y, p1.z;
              count_C = count_list[i];
              normal_3 << p1.normal_x, p1.normal_y, p1.normal_z;
            } else if (l2[1] == l3[1]) {
              C << p2.x, p2.y, p2.z;
              count_C = count_list[pointIdxNKNSearch[m]];
              normal_3 << p2.normal_x, p2.normal_y, p2.normal_z;
            } else {
              C << p3.x, p3.y, p3.z;
              count_C = count_list[pointIdxNKNSearch[n]];
              normal_3 << p3.normal_x, p3.normal_y, p3.normal_z;
            }
            Descriptor single_descriptor;
            single_descriptor.A = A;
            single_descriptor.B = B;
            single_descriptor.C = C;
            single_descriptor.count_A = count_A;
            single_descriptor.count_B = count_B;
            single_descriptor.count_C = count_C;
            single_descriptor.center = (A + B + C) / 3;
            single_descriptor.triangle << scale * a, scale * b, scale * c;
            // single_descriptor.angle[0] = fabs(5 * normal_1.dot(normal_2));
            // single_descriptor.angle[1] = fabs(5 * normal_1.dot(normal_3));
            // single_descriptor.angle[2] = fabs(5 * normal_3.dot(normal_2));
            single_descriptor.angle << 0, 0, 0;
            single_descriptor.frame_number = frame_number;
            single_descriptor.score_frame.push_back(frame_number);
            Eigen::Matrix3d triangle_positon;
            triangle_positon.block<3, 1>(0, 0) = A;
            triangle_positon.block<3, 1>(0, 1) = B;
            triangle_positon.block<3, 1>(0, 2) = C;
            single_descriptor.position_list.push_back(triangle_positon);
            single_descriptor.triangle_scale = scale;
            feat_map[position] = true;
            descriptor_list.push_back(single_descriptor);
          }
        }
      }
    }
  }
}

bool is_line(const std::vector<Eigen::Vector2d> &point_list,
             const double line_threshold) {
  Eigen::Matrix2d covariance = Eigen::Matrix2d::Zero();
  Eigen::Vector2d center = Eigen::Vector2d::Zero();
  for (auto pv : point_list) {
    covariance += pv * pv.transpose();
    center += pv;
  }
  center = center / point_list.size();
  covariance = covariance / point_list.size();
  Eigen::EigenSolver<Eigen::Matrix2d> es(covariance);
  Eigen::Vector2cd evals = es.eigenvalues();
  Eigen::Vector2d evalsReal = evals.real();
  Eigen::Matrix2d::Index evalsMin, evalsMax;
  evalsReal.rowwise().sum().minCoeff(&evalsMin);
  evalsReal.rowwise().sum().maxCoeff(&evalsMax);
  std::cout << " min eigen value:" << evalsReal[evalsMin]
            << " max eigen value:" << evalsReal[evalsMax] << std::endl;
  if (evalsReal[evalsMin] <= line_threshold && evalsReal[evalsMax] >= 0.5) {
    return true;
  } else {
    return false;
  }
}

void extract_corner1(const Eigen::Vector3d &projection_center,
                     const Eigen::Vector3d &projection_normal,
                     const double max_dis_threshold,
                     const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
                     pcl::PointCloud<pcl::PointXYZ>::Ptr &edge_cloud,
                     pcl::PointCloud<pcl::PointXYZ>::Ptr &project_cloud,
                     pcl::PointCloud<pcl::PointXYZINormal>::Ptr &key_points,
                     std::vector<double> &count_list) {
  // old false
  bool max_constraint = false;
  // avia 0.2,0.1,0.6
  double resolution = 0.25;
  double dis_threshold_min = 0.2;
  double dis_threshold_max = 2;
  edge_cloud =
      pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
  project_cloud =
      pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
  key_points = pcl::PointCloud<pcl::PointXYZINormal>::Ptr(
      new pcl::PointCloud<pcl::PointXYZINormal>);
  std::vector<Eigen::Vector3d> key_point_list;
  std::vector<double> key_count_list;
  double A = projection_normal[0];
  double B = projection_normal[1];
  double C = projection_normal[2];
  double D = -(A * projection_center[0] + B * projection_center[1] +
               C * projection_center[2]);
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
  Eigen::Vector3d y_axis = projection_normal.cross(x_axis);
  y_axis.normalize();
  double ax = x_axis[0];
  double bx = x_axis[1];
  double cx = x_axis[2];
  double dx = -(ax * projection_center[0] + bx * projection_center[1] +
                cx * projection_center[2]);
  double ay = y_axis[0];
  double by = y_axis[1];
  double cy = y_axis[2];
  double dy = -(ay * projection_center[0] + by * projection_center[1] +
                cy * projection_center[2]);
  std::vector<Eigen::Vector2d> point_list_2d;
  std::vector<double> dis_list_2d;
  for (size_t i = 0; i < input_cloud->size(); i++) {
    double x = input_cloud->points[i].x;
    double y = input_cloud->points[i].y;
    double z = input_cloud->points[i].z;
    double dis = fabs(x * A + y * B + z * C + D);
    if (dis < dis_threshold_min || dis > dis_threshold_max) {
      // std::cout << "dis:" << dis << std::endl;
      continue;
    } else {
      pcl::PointXYZ pi;
      pi.x = x;
      pi.y = y;
      pi.z = z;
      edge_cloud->points.push_back(pi);
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
    project_cloud->points.push_back(p);
    double project_x =
        cur_project[0] * ay + cur_project[1] * by + cur_project[2] * cy + dy;
    double project_y =
        cur_project[0] * ax + cur_project[1] * bx + cur_project[2] * cx + dx;
    Eigen::Vector2d p_2d(project_x, project_y);
    point_list_2d.push_back(p_2d);
    dis_list_2d.push_back(dis);
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
  // avia 5
  int segmen_base_num = 5;
  double segmen_len = segmen_base_num * resolution;

  int x_segment_num = (max_x - min_x) / segmen_len + 1;
  int y_segment_num = (max_y - min_y) / segmen_len + 1;
  int x_axis_len = (int)((max_x - min_x) / resolution + segmen_base_num);
  int y_axis_len = (int)((max_y - min_y) / resolution + segmen_base_num);
  std::vector<Eigen::Vector2d> img_container[x_axis_len][y_axis_len];
  double img_count[x_axis_len][y_axis_len] = {0};
  double dis_array[x_axis_len][y_axis_len] = {0};
  double gradient_array[x_axis_len][y_axis_len] = {0};
  double mean_x_list[x_axis_len][y_axis_len] = {0};
  double mean_y_list[x_axis_len][y_axis_len] = {0};
  for (int x = 0; x < x_axis_len; x++) {
    for (int y = 0; y < y_axis_len; y++) {
      img_count[x][y] = 0;
      mean_x_list[x][y] = 0;
      mean_y_list[x][y] = 0;
      gradient_array[x][y] = 0;
      dis_array[x][y] = 0;
      std::vector<Eigen::Vector2d> single_container;
      img_container[x][y] = single_container;
    }
  }
  for (size_t i = 0; i < point_list_2d.size(); i++) {
    int x_index = (int)((point_list_2d[i][0] - min_x) / resolution);
    int y_index = (int)((point_list_2d[i][1] - min_y) / resolution);
    mean_x_list[x_index][y_index] += point_list_2d[i][0];
    mean_y_list[x_index][y_index] += point_list_2d[i][1];
    img_count[x_index][y_index]++;
    img_container[x_index][y_index].push_back(point_list_2d[i]);

    if (dis_array[x_index][y_index] == 0) {
      dis_array[x_index][y_index] = dis_list_2d[i];
    } else {
      if (dis_list_2d[i] > dis_array[x_index][y_index]) {
        dis_array[x_index][y_index] = dis_list_2d[i];
      }
    }
  }

  for (int x = 0; x < x_axis_len; x++) {
    for (int y = 0; y < y_axis_len; y++) {
      // calc gradient
      double gradient = 0;
      int cnt = 0;
      int inc = 1;
      for (int x_inc = -inc; x_inc <= inc; x_inc++) {
        for (int y_inc = -inc; y_inc <= inc; y_inc++) {
          int xx = x + x_inc;
          int yy = y + y_inc;
          if (xx >= 0 && xx < x_axis_len && yy >= 0 && yy < y_axis_len) {
            if (xx != x || yy != y) {
              if (img_count[xx][yy] >= 0) {
                gradient += img_count[x][y] - img_count[xx][yy];
                cnt++;
              }
            }
          }
        }
      }
      if (cnt != 0) {
        gradient_array[x][y] = gradient * 1.0 / cnt;
      } else {
        gradient_array[x][y] = 0;
      }
    }
  }

  // for (int x = 0; x < x_axis_len; x++) {
  //   for (int y = 0; y < y_axis_len; y++) {
  //     if (img_count[x][y] <= 0.3 * mean_counts) {
  //       img_count[x][y] = 0;
  //     }
  //   }
  // }
  // debug
  Eigen::Vector3d q_mean(0, 0, 0);
  for (size_t i = 0; i < project_cloud->size(); i++) {
    Eigen::Vector3d qi(project_cloud->points[i].x, project_cloud->points[i].y,
                       project_cloud->points[i].z);
    q_mean += qi;
  }
  q_mean = q_mean / project_cloud->size();
  pcl::PointXYZINormal pi;
  pi.x = q_mean[0];
  pi.y = q_mean[1];
  pi.z = q_mean[2];
  pi.normal_x = projection_normal[0];
  pi.normal_y = projection_normal[1];
  pi.normal_z = projection_normal[2];
  // key_points->push_back(pi);
  // count_list.push_back(10);

  for (int x = 0; x < x_axis_len; x++) {
    for (int y = 0; y < y_axis_len; y++) {
      if (img_count[x][y] > 2) {
        double px = (x + 0.5) * resolution + min_x;
        double py = (y + 0.5) * resolution + min_y;
        // double px = mean_x_list[x][y] / img_count[x][y];
        // double py = mean_y_list[x][y] / img_count[x][y];
        Eigen::Vector3d coord = py * x_axis + px * y_axis + projection_center;
        pcl::PointXYZINormal pi;
        pi.x = coord[0];
        pi.y = coord[1];
        pi.z = coord[2];
        key_points->push_back(pi);
        // count_list.push_back(img_count[x][y]);
        count_list.push_back(dis_array[x][y]);
      }
    }
  }
  return;

  // filter by distance
  std::vector<double> max_dis_list;
  std::vector<int> max_dis_x_index_list;
  std::vector<int> max_dis_y_index_list;

  std::vector<int> max_gradient_list;
  std::vector<int> max_gradient_x_index_list;
  std::vector<int> max_gradient_y_index_list;
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
      if (max_dis >= max_dis_threshold) {
        // std::cout << "max_dis:" << max_dis << " x_index:" << max_dis_x_index
        //           << " y_index:" << max_dis_y_index
        //           << " x_axis_len:" << x_axis_len
        //           << " y_axis_len:" << y_axis_len << std::endl;
        max_dis_list.push_back(max_dis);
        max_dis_x_index_list.push_back(max_dis_x_index);
        max_dis_y_index_list.push_back(max_dis_y_index);
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
    bool is_add = true;
    // for (int j = 0; j < 4; j++) {
    //   Eigen::Vector2i p(max_dis_x_index_list[i], max_dis_y_index_list[i]);
    //   Eigen::Vector2i p1 = p + direction_list[j];
    //   Eigen::Vector2i p2 = p - direction_list[j];
    //   double threshold = dis_array[p[0]][p[1]] * 0.5;
    //   if (dis_array[p1[0]][p1[1]] >= threshold &&
    //       dis_array[p2[0]][p2[1]] >= threshold) {
    //     is_add = false;
    //   } else {
    //     continue;
    //   }
    // }
    for (int dx = -1; dx <= 1; dx++) {
      for (int dy = -1; dy <= 1; dy++) {
        if (dx == 0 && dy == 0) {
          continue;
        }
        Eigen::Vector2i p(max_dis_x_index_list[i], max_dis_y_index_list[i]);
        if (0.8 * dis_array[p[0]][p[1]] < dis_array[p[0] + dx][p[1] + dy]) {
          is_add = false;
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
      Eigen::Vector3d coord = py * x_axis + px * y_axis + projection_center;
      pcl::PointXYZ pi;
      pi.x = coord[0];
      pi.y = coord[1];
      pi.z = coord[2];
      key_point_list.push_back(coord);
      key_count_list.push_back(max_dis_list[i]);
      // key_points->push_back(pi);
      // count_list.push_back(max_gradient_list[i]);
    }
  }
  std::vector<bool> is_add_list;
  for (size_t i = 0; i < key_point_list.size(); i++) {
    is_add_list.push_back(true);
  }
  if (max_constraint) {
    for (size_t i = 0; i < key_point_list.size(); i++) {
      Eigen::Vector3d pi = key_point_list[i];
      for (size_t j = 0; j < key_point_list.size(); j++) {
        Eigen::Vector3d pj = key_point_list[j];
        if (i != j) {
          double dis = sqrt(pow(pi[0] - pj[0], 2) + pow(pi[1] - pj[1], 2) +
                            pow(pi[2] - pj[2], 2));
          if (dis < 1) {
            if (key_count_list[i] > key_count_list[j]) {
              is_add_list[j] = false;
            } else {
              is_add_list[i] = false;
            }
          }
        }
      }
    }
    for (size_t i = 0; i < key_point_list.size(); i++) {
      if (is_add_list[i]) {
        pcl::PointXYZINormal pi;
        pi.x = key_point_list[i][0];
        pi.y = key_point_list[i][1];
        pi.z = key_point_list[i][2];
        pi.normal_x = projection_normal[0];
        pi.normal_y = projection_normal[1];
        pi.normal_z = projection_normal[2];
        key_points->points.push_back(pi);
        count_list.push_back(key_count_list[i]);
      }
    }
  } else {
    for (size_t i = 0; i < key_point_list.size(); i++) {
      pcl::PointXYZINormal pi;
      pi.x = key_point_list[i][0];
      pi.y = key_point_list[i][1];
      pi.z = key_point_list[i][2];
      pi.normal_x = projection_normal[0];
      pi.normal_y = projection_normal[1];
      pi.normal_z = projection_normal[2];
      key_points->points.push_back(pi);
      count_list.push_back(key_count_list[i]);
    }
  }
}

void extract_corner2(const Eigen::Vector3d &projection_center,
                     const Eigen::Vector3d &projection_normal,
                     const double max_count_threshold,
                     const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
                     pcl::PointCloud<pcl::PointXYZ>::Ptr &edge_cloud,
                     pcl::PointCloud<pcl::PointXYZ>::Ptr &project_cloud,
                     pcl::PointCloud<pcl::PointXYZINormal>::Ptr &key_points,
                     std::vector<double> &count_list) {
  // old false
  bool max_constraint = false;
  // avia 0.2,0.1,0.6
  double resolution = 0.25;
  double dis_threshold_min = 0.2;
  double dis_threshold_max = 1;
  edge_cloud =
      pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
  project_cloud =
      pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
  key_points = pcl::PointCloud<pcl::PointXYZINormal>::Ptr(
      new pcl::PointCloud<pcl::PointXYZINormal>);
  std::vector<Eigen::Vector3d> key_point_list;
  std::vector<double> key_count_list;
  double A = projection_normal[0];
  double B = projection_normal[1];
  double C = projection_normal[2];
  double D = -(A * projection_center[0] + B * projection_center[1] +
               C * projection_center[2]);
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
  Eigen::Vector3d y_axis = projection_normal.cross(x_axis);
  y_axis.normalize();
  double ax = x_axis[0];
  double bx = x_axis[1];
  double cx = x_axis[2];
  double dx = -(ax * projection_center[0] + bx * projection_center[1] +
                cx * projection_center[2]);
  double ay = y_axis[0];
  double by = y_axis[1];
  double cy = y_axis[2];
  double dy = -(ay * projection_center[0] + by * projection_center[1] +
                cy * projection_center[2]);
  std::vector<Eigen::Vector2d> point_list_2d;
  for (size_t i = 0; i < input_cloud->size(); i++) {
    double x = input_cloud->points[i].x;
    double y = input_cloud->points[i].y;
    double z = input_cloud->points[i].z;
    double dis = fabs(x * A + y * B + z * C + D);
    if (dis < dis_threshold_min || dis > dis_threshold_max) {
      // std::cout << "dis:" << dis << std::endl;
      continue;
    } else {
      pcl::PointXYZ pi;
      pi.x = x;
      pi.y = y;
      pi.z = z;
      edge_cloud->points.push_back(pi);
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
    project_cloud->points.push_back(p);
    double project_x =
        cur_project[0] * ay + cur_project[1] * by + cur_project[2] * cy + dy;
    double project_y =
        cur_project[0] * ax + cur_project[1] * bx + cur_project[2] * cx + dx;
    Eigen::Vector2d p_2d(project_x, project_y);
    point_list_2d.push_back(p_2d);
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
  // avia 5
  int segmen_base_num = 3;
  double segmen_len = segmen_base_num * resolution;

  int x_segment_num = (max_x - min_x) / segmen_len + 1;
  int y_segment_num = (max_y - min_y) / segmen_len + 1;
  int x_axis_len = (int)((max_x - min_x) / resolution + segmen_base_num);
  int y_axis_len = (int)((max_y - min_y) / resolution + segmen_base_num);
  std::vector<Eigen::Vector2d> img_container[x_axis_len][y_axis_len];
  double img_count[x_axis_len][y_axis_len] = {0};
  double gradient_array[x_axis_len][y_axis_len] = {0};
  double mean_x_list[x_axis_len][y_axis_len] = {0};
  double mean_y_list[x_axis_len][y_axis_len] = {0};
  for (int x = 0; x < x_axis_len; x++) {
    for (int y = 0; y < y_axis_len; y++) {
      img_count[x][y] = 0;
      mean_x_list[x][y] = 0;
      mean_y_list[x][y] = 0;
      gradient_array[x][y] = 0;
      std::vector<Eigen::Vector2d> single_container;
      img_container[x][y] = single_container;
    }
  }
  for (size_t i = 0; i < point_list_2d.size(); i++) {
    int x_index = (int)((point_list_2d[i][0] - min_x) / resolution);
    int y_index = (int)((point_list_2d[i][1] - min_y) / resolution);
    mean_x_list[x_index][y_index] += point_list_2d[i][0];
    mean_y_list[x_index][y_index] += point_list_2d[i][1];
    img_count[x_index][y_index]++;
    img_container[x_index][y_index].push_back(point_list_2d[i]);
  }
  for (int x = 0; x < x_axis_len; x++) {
    for (int y = 0; y < y_axis_len; y++) {
      // calc gradient
      double gradient = 0;
      int cnt = 0;
      int inc = 1;
      for (int x_inc = -inc; x_inc <= inc; x_inc++) {
        for (int y_inc = -inc; y_inc <= inc; y_inc++) {
          int xx = x + x_inc;
          int yy = y + y_inc;
          if (xx >= 0 && xx < x_axis_len && yy >= 0 && yy < y_axis_len) {
            if (xx != x || yy != y) {
              if (img_count[xx][yy] >= 0) {
                gradient += img_count[x][y] - img_count[xx][yy];
                cnt++;
              }
            }
          }
        }
      }
      if (cnt != 0) {
        gradient_array[x][y] = gradient * 1.0 / cnt;
      } else {
        gradient_array[x][y] = 0;
      }
    }
  }

  // for (int x = 0; x < x_axis_len; x++) {
  //   for (int y = 0; y < y_axis_len; y++) {
  //     if (img_count[x][y] <= 0.3 * mean_counts) {
  //       img_count[x][y] = 0;
  //     }
  //   }
  // }
  // debug
  // Eigen::Vector3d q_mean(0, 0, 0);
  // for (size_t i = 0; i < project_cloud->size(); i++) {
  //   Eigen::Vector3d qi(project_cloud->points[i].x,
  //   project_cloud->points[i].y,
  //                      project_cloud->points[i].z);
  //   q_mean += qi;
  // }
  // q_mean = q_mean / project_cloud->size();
  // pcl::PointXYZINormal pi;
  // pi.x = q_mean[0];
  // pi.y = q_mean[1];
  // pi.z = q_mean[2];
  // pi.normal_x = projection_normal[0];
  // pi.normal_y = projection_normal[1];
  // pi.normal_z = projection_normal[2];
  // key_points->push_back(pi);
  // count_list.push_back(10);

  // for (int x = 0; x < x_axis_len; x++) {
  //   for (int y = 0; y < y_axis_len; y++) {
  //     if (img_count[x][y] > 2) {
  //       double px = (x + 0.5) * resolution + min_x;
  //       double py = (y + 0.5) * resolution + min_y;
  //       // double px = mean_x_list[x][y] / img_count[x][y];
  //       // double py = mean_y_list[x][y] / img_count[x][y];
  //       Eigen::Vector3d coord = py * x_axis + px * y_axis +
  //       projection_center; pcl::PointXYZINormal pi; pi.x = coord[0]; pi.y =
  //       coord[1]; pi.z = coord[2]; key_points->push_back(pi);
  //       //        count_list.push_back(img_count[x][y]);
  //       count_list.push_back(img_count[x][y]);
  //     }
  //   }
  // }
  // return;

  // filter by gradient
  std::vector<int> max_gradient_list;
  std::vector<int> max_gradient_x_index_list;
  std::vector<int> max_gradient_y_index_list;
  for (int x_segment_index = 0; x_segment_index < x_segment_num;
       x_segment_index++) {
    for (int y_segment_index = 0; y_segment_index < y_segment_num;
         y_segment_index++) {
      double max_gradient = 0;
      int max_gradient_x_index = -10;
      int max_gradient_y_index = -10;
      for (int x_index = x_segment_index * segmen_base_num;
           x_index < (x_segment_index + 1) * segmen_base_num; x_index++) {
        for (int y_index = y_segment_index * segmen_base_num;
             y_index < (y_segment_index + 1) * segmen_base_num; y_index++) {
          if (img_count[x_index][y_index] > max_gradient) {
            max_gradient = img_count[x_index][y_index];
            max_gradient_x_index = x_index;
            max_gradient_y_index = y_index;
          }
        }
      }
      if (max_gradient >= max_count_threshold) {
        // std::cout << "max_count:" << max_gradient
        //           << " x_index:" << max_gradient_x_index
        //           << " y_index:" << max_gradient_y_index
        //           << " x_axis_len:" << x_axis_len
        //           << " y_axis_len:" << y_axis_len << std::endl;
        max_gradient_list.push_back(max_gradient);
        max_gradient_x_index_list.push_back(max_gradient_x_index);
        max_gradient_y_index_list.push_back(max_gradient_y_index);
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
  for (size_t i = 0; i < max_gradient_list.size(); i++) {
    bool is_add = true;
    for (int j = 0; j < 4; j++) {
      Eigen::Vector2i p(max_gradient_x_index_list[i],
                        max_gradient_y_index_list[i]);
      Eigen::Vector2i p1 = p + direction_list[j];
      Eigen::Vector2i p2 = p - direction_list[j];
      int threshold = img_count[p[0]][p[1]] / 2;
      if (img_count[p1[0]][p1[1]] >= threshold &&
          img_count[p2[0]][p2[1]] >= threshold) {
        is_add = false;
      } else {
        continue;
      }
    }
    if (is_add) {
      double px =
          mean_x_list[max_gradient_x_index_list[i]]
                     [max_gradient_y_index_list[i]] /
          img_count[max_gradient_x_index_list[i]][max_gradient_y_index_list[i]];
      double py =
          mean_y_list[max_gradient_x_index_list[i]]
                     [max_gradient_y_index_list[i]] /
          img_count[max_gradient_x_index_list[i]][max_gradient_y_index_list[i]];
      // std::cout << "max_count_x_index_list[i]: " <<
      // max_gradient_x_index_list[i]
      //           << std::endl;
      // std::cout << "px,py " << px << "," << py << std::endl;
      Eigen::Vector3d coord = py * x_axis + px * y_axis + projection_center;
      pcl::PointXYZ pi;
      pi.x = coord[0];
      pi.y = coord[1];
      pi.z = coord[2];
      key_point_list.push_back(coord);
      key_count_list.push_back(max_gradient_list[i]);
      // key_points->push_back(pi);
      // count_list.push_back(max_gradient_list[i]);
    }
  }
  std::vector<bool> is_add_list;
  for (size_t i = 0; i < key_point_list.size(); i++) {
    is_add_list.push_back(true);
  }
  if (max_constraint) {
    for (size_t i = 0; i < key_point_list.size(); i++) {
      Eigen::Vector3d pi = key_point_list[i];
      for (size_t j = 0; j < key_point_list.size(); j++) {
        Eigen::Vector3d pj = key_point_list[j];
        if (i != j) {
          double dis = sqrt(pow(pi[0] - pj[0], 2) + pow(pi[1] - pj[1], 2) +
                            pow(pi[2] - pj[2], 2));
          if (dis < 1) {
            if (key_count_list[i] > key_count_list[j]) {
              is_add_list[j] = false;
            } else {
              is_add_list[i] = false;
            }
          }
        }
      }
    }
    for (size_t i = 0; i < key_point_list.size(); i++) {
      if (is_add_list[i]) {
        pcl::PointXYZINormal pi;
        pi.x = key_point_list[i][0];
        pi.y = key_point_list[i][1];
        pi.z = key_point_list[i][2];
        pi.normal_x = projection_normal[0];
        pi.normal_y = projection_normal[1];
        pi.normal_z = projection_normal[2];
        key_points->points.push_back(pi);
        count_list.push_back(key_count_list[i]);
      }
    }
  } else {
    for (size_t i = 0; i < key_point_list.size(); i++) {
      pcl::PointXYZINormal pi;
      pi.x = key_point_list[i][0];
      pi.y = key_point_list[i][1];
      pi.z = key_point_list[i][2];
      pi.normal_x = projection_normal[0];
      pi.normal_y = projection_normal[1];
      pi.normal_z = projection_normal[2];
      key_points->points.push_back(pi);
      count_list.push_back(key_count_list[i]);
    }
  }
}

void extract_corner3(const Eigen::Vector3d &projection_center,
                     const Eigen::Vector3d &projection_normal,
                     const double max_dis_threshold,
                     const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
                     pcl::PointCloud<pcl::PointXYZ>::Ptr &edge_cloud,
                     pcl::PointCloud<pcl::PointXYZ>::Ptr &project_cloud,
                     pcl::PointCloud<pcl::PointXYZINormal>::Ptr &key_points,
                     std::vector<double> &count_list) {
  // old false
  bool max_constraint = false;
  // avia 0.2,0.1,0.6
  double resolution = 0.5;
  double dis_threshold_min = 0.2;
  double dis_threshold_max = 3;
  double high_inc = 0.1;
  edge_cloud =
      pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
  project_cloud =
      pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
  key_points = pcl::PointCloud<pcl::PointXYZINormal>::Ptr(
      new pcl::PointCloud<pcl::PointXYZINormal>);
  std::vector<Eigen::Vector3d> key_point_list;
  std::vector<double> key_count_list;
  double A = projection_normal[0];
  double B = projection_normal[1];
  double C = projection_normal[2];
  double D = -(A * projection_center[0] + B * projection_center[1] +
               C * projection_center[2]);
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
  Eigen::Vector3d y_axis = projection_normal.cross(x_axis);
  y_axis.normalize();
  double ax = x_axis[0];
  double bx = x_axis[1];
  double cx = x_axis[2];
  double dx = -(ax * projection_center[0] + bx * projection_center[1] +
                cx * projection_center[2]);
  double ay = y_axis[0];
  double by = y_axis[1];
  double cy = y_axis[2];
  double dy = -(ay * projection_center[0] + by * projection_center[1] +
                cy * projection_center[2]);
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
      // std::cout << "dis:" << dis << std::endl;
      continue;
    } else {
      if (dis > dis_threshold_min && dis <= dis_threshold_max) {
        pi.x = x;
        pi.y = y;
        pi.z = z;
        // edge_cloud->points.push_back(pi);
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
    project_cloud->points.push_back(p);
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
  // avia 5
  int segmen_base_num = 3;
  double segmen_len = segmen_base_num * resolution;

  int x_segment_num = (max_x - min_x) / segmen_len + 1;
  int y_segment_num = (max_y - min_y) / segmen_len + 1;
  int x_axis_len = (int)((max_x - min_x) / resolution + segmen_base_num);
  int y_axis_len = (int)((max_y - min_y) / resolution + segmen_base_num);
  std::vector<Eigen::Vector2d> img_container[x_axis_len][y_axis_len];
  std::vector<double> dis_container[x_axis_len][y_axis_len];
  pcl::PointCloud<pcl::PointXYZ> edge_cloud_array[x_axis_len][y_axis_len];
  double img_count[x_axis_len][y_axis_len] = {0};
  double dis_array[x_axis_len][y_axis_len] = {0};
  double gradient_array[x_axis_len][y_axis_len] = {0};
  double mean_x_list[x_axis_len][y_axis_len] = {0};
  double mean_y_list[x_axis_len][y_axis_len] = {0};
  for (int x = 0; x < x_axis_len; x++) {
    for (int y = 0; y < y_axis_len; y++) {
      img_count[x][y] = 0;
      mean_x_list[x][y] = 0;
      mean_y_list[x][y] = 0;
      gradient_array[x][y] = 0;
      dis_array[x][y] = 0;
      std::vector<Eigen::Vector2d> single_container;
      img_container[x][y] = single_container;
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
    img_container[x_index][y_index].push_back(point_list_2d[i]);
    dis_container[x_index][y_index].push_back(dis_list_2d[i]);
    edge_cloud_array[x_index][y_index].push_back(point_list_3d.points[i]);
    if (dis_array[x_index][y_index] == 0) {
      dis_array[x_index][y_index] = dis_list_2d[i];
    } else {
      if (dis_list_2d[i] > dis_array[x_index][y_index]) {
        dis_array[x_index][y_index] = dis_list_2d[i];
      }
    }
  }

  for (int x = 0; x < x_axis_len; x++) {
    for (int y = 0; y < y_axis_len; y++) {
      // calc segment dis array
      if (img_count[x][y] > 0) {
        int cut_num = (dis_threshold_max - dis_threshold_min) / high_inc;
        std::vector<double> cnt_list;
        for (size_t i = 0; i < cut_num; i++) {
          cnt_list.push_back(0);
        }
        for (size_t j = 0; j < dis_container[x][y].size(); j++) {
          int cnt_index =
              (dis_container[x][y][j] - dis_threshold_min) / high_inc;
          cnt_list[cnt_index]++;
        }
        double segmnt_dis = 0;
        for (size_t i = 0; i < cut_num; i++) {
          if (cnt_list[i] >= 2) {
            segmnt_dis++;
          }
        }
        dis_array[x][y] = segmnt_dis;
      }
    }
  }

  // for (int x = 0; x < x_axis_len; x++) {
  //   for (int y = 0; y < y_axis_len; y++) {
  //     if (img_count[x][y] <= 0.3 * mean_counts) {
  //       img_count[x][y] = 0;
  //     }
  //   }
  // }
  // debug
  // Eigen::Vector3d q_mean(0, 0, 0);
  // for (size_t i = 0; i < project_cloud->size(); i++) {
  //   Eigen::Vector3d qi(project_cloud->points[i].x,
  //   project_cloud->points[i].y,
  //                      project_cloud->points[i].z);
  //   q_mean += qi;
  // }
  // q_mean = q_mean / project_cloud->size();
  // pcl::PointXYZINormal pi;
  // pi.x = q_mean[0];
  // pi.y = q_mean[1];
  // pi.z = q_mean[2];
  // pi.normal_x = projection_normal[0];
  // pi.normal_y = projection_normal[1];
  // pi.normal_z = projection_normal[2];
  // key_points->push_back(pi);
  // count_list.push_back(10);

  // for (int x = 0; x < x_axis_len; x++) {
  //   for (int y = 0; y < y_axis_len; y++) {
  //     if (img_count[x][y] > 0) {
  //       double px = (x + 0.5) * resolution + min_x;
  //       double py = (y + 0.5) * resolution + min_y;
  //       // double px = mean_x_list[x][y] / img_count[x][y];
  //       // double py = mean_y_list[x][y] / img_count[x][y];
  //       Eigen::Vector3d coord = py * x_axis + px * y_axis +
  //       projection_center; pcl::PointXYZINormal pi; pi.x = coord[0]; pi.y =
  //       coord[1]; pi.z = coord[2]; key_points->push_back(pi);
  //       //        count_list.push_back(img_count[x][y]);
  //       int upper_cnt = 0;
  //       int lower_cnt = 0;
  //       int cut_num = (dis_threshold_max - dis_threshold_min) / high_inc;
  //       std::vector<double> cnt_list;
  //       for (size_t i = 0; i < cut_num; i++) {
  //         cnt_list.push_back(0);
  //       }
  //       for (size_t j = 0; j < dis_container[x][y].size(); j++) {
  //         int cnt_index =
  //             (dis_container[x][y][j] - dis_threshold_min) / high_inc;
  //         cnt_list[cnt_index]++;
  //       }
  //       // for (size_t j = 0; j < dis_container[x][y].size(); j++) {
  //       //   if (dis_container[x][y][j] > max_dis_threshold &&
  //       //       dis_container[x][y][j] < max_dis_threshold + high_inc) {
  //       //     upper_cnt++;
  //       //   }
  //       //   if (dis_container[x][y][j] < max_dis_threshold) {
  //       //     lower_cnt++;
  //       //   }
  //       // }
  //       double count = 0;
  //       for (size_t i = 0; i < cut_num; i++) {
  //         if (cnt_list[i] >= 1) {
  //           count++;
  //         }
  //       }
  //       count_list.push_back(count);
  //       // if (upper_cnt >= 3) {
  //       //   if (lower_cnt >= 3) {
  //       //     count_list.push_back(1);
  //       //   } else {
  //       //     count_list.push_back(2);
  //       //   }
  //       // } else {
  //       //   count_list.push_back(0);
  //       // }
  //     }
  //     //  else {
  //     //   double px = (x + 0.5) * resolution + min_x;
  //     //   double py = (y + 0.5) * resolution + min_y;
  //     //   // double px = mean_x_list[x][y] / img_count[x][y];
  //     //   // double py = mean_y_list[x][y] / img_count[x][y];
  //     //   Eigen::Vector3d coord = py * x_axis + px * y_axis +
  //     //   projection_center; pcl::PointXYZINormal pi; pi.x = coord[0]; pi.y
  //     =
  //     //   coord[1]; pi.z = coord[2]; key_points->push_back(pi);
  //     //   count_list.push_back(-1);
  //     // }
  //   }
  // }
  // return;

  // filter by distance
  std::vector<double> max_dis_list;
  std::vector<int> max_dis_x_index_list;
  std::vector<int> max_dis_y_index_list;

  std::vector<int> max_gradient_list;
  std::vector<int> max_gradient_x_index_list;
  std::vector<int> max_gradient_y_index_list;
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
      if (max_dis >= 4) {
        // std::cout << "max_dis:" << max_dis << " x_index:" << max_dis_x_index
        //           << " y_index:" << max_dis_y_index
        //           << " x_axis_len:" << x_axis_len
        //           << " y_axis_len:" << y_axis_len << std::endl;
        max_dis_list.push_back(max_dis);
        max_dis_x_index_list.push_back(max_dis_x_index);
        max_dis_y_index_list.push_back(max_dis_y_index);
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
    bool is_add = true;
    for (int j = 0; j < 4; j++) {
      Eigen::Vector2i p(max_dis_x_index_list[i], max_dis_y_index_list[i]);
      Eigen::Vector2i p1 = p + direction_list[j];
      Eigen::Vector2i p2 = p - direction_list[j];
      double threshold = dis_array[p[0]][p[1]] - 2;
      if (dis_array[p1[0]][p1[1]] >= threshold ||
          dis_array[p2[0]][p2[1]] >= threshold) {
        // double px = mean_x_list[p1[0]][p1[1]] / img_count[p1[0]][p1[1]];
        // double py = mean_y_list[p1[0]][p1[1]] / img_count[p1[0]][p1[1]];
        // // std::cout << "max dis: " << max_dis_list[i] << std::endl;
        // // std::cout << "px,py " << px << "," << py << std::endl;
        // Eigen::Vector3d coord = py * x_axis + px * y_axis +
        // projection_center; key_point_list.push_back(coord);
        // key_count_list.push_back(dis_array[p1[0]][p1[1]]);
        // px = mean_x_list[p2[0]][p2[1]] / img_count[p2[0]][p2[1]];
        // py = mean_y_list[p2[0]][p2[1]] / img_count[p2[0]][p2[1]];
        // coord = py * x_axis + px * y_axis + projection_center;
        // key_point_list.push_back(coord);
        // key_count_list.push_back(dis_array[p2[0]][p2[1]]);
        is_add = true;
      } else {
        continue;
      }
    }
    for (int dx = -1; dx <= 1; dx++) {
      for (int dy = -1; dy <= 1; dy++) {
        if (dx == 0 && dy == 0) {
          continue;
        }
        Eigen::Vector2i p(max_dis_x_index_list[i], max_dis_y_index_list[i]);
        if (0.8 * dis_array[p[0]][p[1]] < dis_array[p[0] + dx][p[1] + dy]) {
          is_add = false;
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
      // std::cout << "max dis: " << max_dis_list[i] << std::endl;
      // std::cout << "px,py " << px << "," << py << std::endl;
      Eigen::Vector3d coord = py * x_axis + px * y_axis + projection_center;
      pcl::PointXYZ pi;
      pi.x = coord[0];
      pi.y = coord[1];
      pi.z = coord[2];
      key_point_list.push_back(coord);
      key_count_list.push_back(max_dis_list[i]);
      for (size_t j = 0;
           j <
           edge_cloud_array[max_dis_x_index_list[i]][max_dis_y_index_list[i]]
               .size();
           j++) {
        edge_cloud->points.push_back(
            edge_cloud_array[max_dis_x_index_list[i]][max_dis_y_index_list[i]]
                .points[j]);
      }

      // key_points->push_back(pi);
      // count_list.push_back(max_gradient_list[i]);
    }
  }
  std::vector<bool> is_add_list;
  for (size_t i = 0; i < key_point_list.size(); i++) {
    is_add_list.push_back(true);
  }
  if (max_constraint) {
    for (size_t i = 0; i < key_point_list.size(); i++) {
      Eigen::Vector3d pi = key_point_list[i];
      for (size_t j = 0; j < key_point_list.size(); j++) {
        Eigen::Vector3d pj = key_point_list[j];
        if (i != j) {
          double dis = sqrt(pow(pi[0] - pj[0], 2) + pow(pi[1] - pj[1], 2) +
                            pow(pi[2] - pj[2], 2));
          if (dis < 1) {
            if (key_count_list[i] > key_count_list[j]) {
              is_add_list[j] = false;
            } else {
              is_add_list[i] = false;
            }
          }
        }
      }
    }
    for (size_t i = 0; i < key_point_list.size(); i++) {
      if (is_add_list[i]) {
        pcl::PointXYZINormal pi;
        pi.x = key_point_list[i][0];
        pi.y = key_point_list[i][1];
        pi.z = key_point_list[i][2];
        pi.normal_x = projection_normal[0];
        pi.normal_y = projection_normal[1];
        pi.normal_z = projection_normal[2];
        key_points->points.push_back(pi);
        count_list.push_back(key_count_list[i]);
      }
    }
  } else {
    for (size_t i = 0; i < key_point_list.size(); i++) {
      pcl::PointXYZINormal pi;
      pi.x = key_point_list[i][0];
      pi.y = key_point_list[i][1];
      pi.z = key_point_list[i][2];
      pi.normal_x = projection_normal[0];
      pi.normal_y = projection_normal[1];
      pi.normal_z = projection_normal[2];
      key_points->points.push_back(pi);
      count_list.push_back(key_count_list[i]);
    }
  }
}

void extract_binary(const Eigen::Vector3d &projection_center,
                    const Eigen::Vector3d &projection_normal,
                    const double min_count_threshold,
                    const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
                    const bool is_debug,
                    std::vector<BinaryDescriptor> &binary_list) {
  binary_list.clear();
  // avia 0.2,0.1,0.6
  double corner_min_dis = min_count_threshold;
  // old 0.5
  double resolution = 0.5;
  double dis_threshold_min = 0.2;
  double dis_threshold_max = 5;
  // old 0.2
  double high_inc = 0.2;
  double A = projection_normal[0];
  double B = projection_normal[1];
  double C = projection_normal[2];
  double D = -(A * projection_center[0] + B * projection_center[1] +
               C * projection_center[2]);
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
  Eigen::Vector3d y_axis = projection_normal.cross(x_axis);
  y_axis.normalize();
  double ax = x_axis[0];
  double bx = x_axis[1];
  double cx = x_axis[2];
  double dx = -(ax * projection_center[0] + bx * projection_center[1] +
                cx * projection_center[2]);
  double ay = y_axis[0];
  double by = y_axis[1];
  double cy = y_axis[2];
  double dy = -(ay * projection_center[0] + by * projection_center[1] +
                cy * projection_center[2]);

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
      // std::cout << "dis:" << dis << std::endl;
      continue;
    } else {
      if (dis > dis_threshold_min && dis <= dis_threshold_max) {
        pi.x = x;
        pi.y = y;
        pi.z = z;
        // edge_cloud->points.push_back(pi);
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
  // avia 5
  int segmen_base_num = 5;
  double segmen_len = segmen_base_num * resolution;

  int x_segment_num = (max_x - min_x) / segmen_len + 1;
  int y_segment_num = (max_y - min_y) / segmen_len + 1;
  int x_axis_len = (int)((max_x - min_x) / resolution + segmen_base_num);
  int y_axis_len = (int)((max_y - min_y) / resolution + segmen_base_num);

  std::vector<float> **dis_container = new std::vector<float> *[x_axis_len];
  BinaryDescriptor **binary_container = new BinaryDescriptor *[x_axis_len];
  for (int i = 0; i < x_axis_len; i++) {
    dis_container[i] = new std::vector<float>[y_axis_len];
    binary_container[i] = new BinaryDescriptor[y_axis_len];
  }
  // std::vector<float> dis_container[x_axis_len][y_axis_len];
  // pcl::PointCloud<pcl::PointXYZ> edge_cloud_array[x_axis_len][y_axis_len];
  float **img_count = new float *[x_axis_len];
  for (int i = 0; i < x_axis_len; i++) {
    img_count[i] = new float[y_axis_len];
  }
  float **dis_array = new float *[x_axis_len];
  for (int i = 0; i < x_axis_len; i++) {
    dis_array[i] = new float[y_axis_len];
  }
  float **mean_x_list = new float *[x_axis_len];
  for (int i = 0; i < x_axis_len; i++) {
    mean_x_list[i] = new float[y_axis_len];
  }
  float **mean_y_list = new float *[x_axis_len];
  for (int i = 0; i < x_axis_len; i++) {
    mean_y_list[i] = new float[y_axis_len];
  }
  // float img_count[x_axis_len][y_axis_len] = {0};
  // float dis_array[x_axis_len][y_axis_len] = {0};
  // float mean_x_list[x_axis_len][y_axis_len] = {0};
  // float mean_y_list[x_axis_len][y_axis_len] = {0};
  for (int x = 0; x < x_axis_len; x++) {
    for (int y = 0; y < y_axis_len; y++) {
      img_count[x][y] = 0;
      mean_x_list[x][y] = 0;
      mean_y_list[x][y] = 0;
      dis_array[x][y] = 0;
      std::vector<float> single_dis_container;
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
        single_binary.occupy_array = occup_list;
        single_binary.summary = segmnt_dis;
        binary_container[x][y] = single_binary;
        // for debug
        if (is_debug) {
          double px = mean_x_list[x][y] / img_count[x][y];
          double py = mean_y_list[x][y] / img_count[x][y];
          Eigen::Vector3d coord = py * x_axis + px * y_axis + projection_center;
          pcl::PointXYZ pi;
          pi.x = coord[0];
          pi.y = coord[1];
          pi.z = coord[2];
          single_binary.location = coord;
          binary_list.push_back(single_binary);
        }
      }
    }
  }
  if (is_debug)
    return;
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
      if (max_dis >= corner_min_dis) {
        bool is_touch = true;
        is_touch =
            binary_container[max_dis_x_index][max_dis_y_index]
                .occupy_array[0] ||
            binary_container[max_dis_x_index][max_dis_y_index]
                .occupy_array[1] ||
            binary_container[max_dis_x_index][max_dis_y_index]
                .occupy_array[2] ||
            binary_container[max_dis_x_index][max_dis_y_index].occupy_array[3];
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
    bool is_add = true;
    for (int j = 0; j < 4; j++) {
      Eigen::Vector2i p(max_dis_x_index_list[i], max_dis_y_index_list[i]);
      if (p[0] <= 0 || p[0] >= x_axis_len - 1 || p[1] <= 0 ||
          p[1] >= y_axis_len - 1) {
        continue;
      }
      Eigen::Vector2i p1 = p + direction_list[j];
      Eigen::Vector2i p2 = p - direction_list[j];
      double threshold = dis_array[p[0]][p[1]] - 3;
      // if (dis_array[p1[0]][p1[1]] >= threshold) {
      //   if (dis_array[p2[0]][p2[1]] > 0.5 * dis_array[p[0]][p[1]]) {
      //     is_add = false;
      //   }
      // }
      // if (dis_array[p2[0]][p2[1]] >= threshold) {
      //   if (dis_array[p1[0]][p1[1]] > 0.5 * dis_array[p[0]][p[1]]) {
      //     is_add = false;
      //   }
      // }
      // if (dis_array[p1[0]][p1[1]] >= threshold) {
      //   if (dis_array[p2[0]][p2[1]] >= threshold) {
      //     is_add = false;
      //   }
      // }
      // if (dis_array[p2[0]][p2[1]] >= threshold) {
      //   if (dis_array[p1[0]][p1[1]] >= threshold) {
      //     is_add = false;
      //   }
      // }
    }
    if (is_add) {
      double px =
          mean_x_list[max_dis_x_index_list[i]][max_dis_y_index_list[i]] /
          img_count[max_dis_x_index_list[i]][max_dis_y_index_list[i]];
      double py =
          mean_y_list[max_dis_x_index_list[i]][max_dis_y_index_list[i]] /
          img_count[max_dis_x_index_list[i]][max_dis_y_index_list[i]];
      Eigen::Vector3d coord = py * x_axis + px * y_axis + projection_center;
      pcl::PointXYZ pi;
      pi.x = coord[0];
      pi.y = coord[1];
      pi.z = coord[2];
      BinaryDescriptor single_binary =
          binary_container[max_dis_x_index_list[i]][max_dis_y_index_list[i]];
      single_binary.location = coord;
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

void extract_corner4(const Eigen::Vector3d &projection_center,
                     const Eigen::Vector3d &projection_normal,
                     const double max_dis_threshold,
                     const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
                     pcl::PointCloud<pcl::PointXYZ>::Ptr &edge_cloud,
                     pcl::PointCloud<pcl::PointXYZ>::Ptr &project_cloud,
                     pcl::PointCloud<pcl::PointXYZINormal>::Ptr &key_points,
                     std::vector<double> &count_list) {

  // old false
  bool max_constraint = false;
  // avia 0.2,0.1,0.6
  double corner_min_dis = 6;
  double resolution = 0.5;
  double dis_threshold_min = 0.2;
  double dis_threshold_max = 5;
  double high_inc = 0.2;
  edge_cloud =
      pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
  project_cloud =
      pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
  key_points = pcl::PointCloud<pcl::PointXYZINormal>::Ptr(
      new pcl::PointCloud<pcl::PointXYZINormal>);
  std::vector<Eigen::Vector3d> key_point_list;
  std::vector<double> key_count_list;
  double A = projection_normal[0];
  double B = projection_normal[1];
  double C = projection_normal[2];
  double D = -(A * projection_center[0] + B * projection_center[1] +
               C * projection_center[2]);
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
  Eigen::Vector3d y_axis = projection_normal.cross(x_axis);
  y_axis.normalize();
  double ax = x_axis[0];
  double bx = x_axis[1];
  double cx = x_axis[2];
  double dx = -(ax * projection_center[0] + bx * projection_center[1] +
                cx * projection_center[2]);
  double ay = y_axis[0];
  double by = y_axis[1];
  double cy = y_axis[2];
  double dy = -(ay * projection_center[0] + by * projection_center[1] +
                cy * projection_center[2]);
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
      // std::cout << "dis:" << dis << std::endl;
      continue;
    } else {
      if (dis > dis_threshold_min && dis <= dis_threshold_max) {
        pi.x = x;
        pi.y = y;
        pi.z = z;
        // edge_cloud->points.push_back(pi);
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
    project_cloud->points.push_back(p);
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
  std::cout << "min x:" << min_x << ", max x:" << max_x << ", min y:" << min_y
            << ", max y:" << max_y << std::endl;
  // segment project cloud
  // avia 5
  int segmen_base_num = 5;
  double segmen_len = segmen_base_num * resolution;

  int x_segment_num = (max_x - min_x) / segmen_len + 1;
  int y_segment_num = (max_y - min_y) / segmen_len + 1;
  int x_axis_len = (int)((max_x - min_x) / resolution + segmen_base_num);
  int y_axis_len = (int)((max_y - min_y) / resolution + segmen_base_num);

  std::vector<double> dis_container[x_axis_len][y_axis_len];

  // pcl::PointCloud<pcl::PointXYZ> edge_cloud_array[x_axis_len][y_axis_len];
  float img_count[x_axis_len][y_axis_len] = {0};
  float dis_array[x_axis_len][y_axis_len] = {0};
  float mean_x_list[x_axis_len][y_axis_len] = {0};
  float mean_y_list[x_axis_len][y_axis_len] = {0};
  std::cout << "x axis len:" << x_axis_len << ", y axis len:" << y_axis_len
            << std::endl;

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
    // edge_cloud_array[x_index][y_index].push_back(point_list_3d.points[i]);
  }
  for (int x = 0; x < x_axis_len; x++) {
    for (int y = 0; y < y_axis_len; y++) {
      // calc segment dis array
      if (img_count[x][y] > 0) {
        int cut_num = (dis_threshold_max - dis_threshold_min) / high_inc;
        std::vector<double> cnt_list;
        for (size_t i = 0; i < cut_num; i++) {
          cnt_list.push_back(0);
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
          }
        }
        dis_array[x][y] = segmnt_dis;
      }
    }
  }
  // filter by distance
  std::vector<double> max_dis_list;
  std::vector<int> max_dis_x_index_list;
  std::vector<int> max_dis_y_index_list;

  std::vector<int> max_gradient_list;
  std::vector<int> max_gradient_x_index_list;
  std::vector<int> max_gradient_y_index_list;
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
      if (max_dis >= corner_min_dis) {
        // std::cout << "max_dis:" << max_dis << " x_index:" << max_dis_x_index
        //           << " y_index:" << max_dis_y_index
        //           << " x_axis_len:" << x_axis_len
        //           << " y_axis_len:" << y_axis_len << std::endl;
        max_dis_list.push_back(max_dis);
        max_dis_x_index_list.push_back(max_dis_x_index);
        max_dis_y_index_list.push_back(max_dis_y_index);
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
    bool is_add = true;
    for (int j = 0; j < 4; j++) {
      Eigen::Vector2i p(max_dis_x_index_list[i], max_dis_y_index_list[i]);
      Eigen::Vector2i p1 = p + direction_list[j];
      Eigen::Vector2i p2 = p - direction_list[j];
      double threshold = dis_array[p[0]][p[1]] - 3;
      if (dis_array[p1[0]][p1[1]] >= threshold) {
        if (dis_array[p2[0]][p2[1]] > 0.5 * dis_array[p[0]][p[1]]) {
          is_add = false;
        }
      }
      if (dis_array[p2[0]][p2[1]] >= threshold) {
        if (dis_array[p1[0]][p1[1]] > 0.5 * dis_array[p[0]][p[1]]) {
          is_add = false;
        }
      }
      // if (dis_array[p1[0]][p1[1]] >= threshold ||
      //     dis_array[p2[0]][p2[1]] >= threshold) {
      //   // double px = mean_x_list[p1[0]][p1[1]] / img_count[p1[0]][p1[1]];
      //   // double py = mean_y_list[p1[0]][p1[1]] / img_count[p1[0]][p1[1]];
      //   // // std::cout << "max dis: " << max_dis_list[i] << std::endl;
      //   // // std::cout << "px,py " << px << "," << py << std::endl;
      //   // Eigen::Vector3d coord = py * x_axis + px * y_axis +
      //   // projection_center; key_point_list.push_back(coord);
      //   // key_count_list.push_back(dis_array[p1[0]][p1[1]]);
      //   // px = mean_x_list[p2[0]][p2[1]] / img_count[p2[0]][p2[1]];
      //   // py = mean_y_list[p2[0]][p2[1]] / img_count[p2[0]][p2[1]];
      //   // coord = py * x_axis + px * y_axis + projection_center;
      //   // key_point_list.push_back(coord);
      //   // key_count_list.push_back(dis_array[p2[0]][p2[1]]);
      //   is_add = false;
      // } else {
      //   continue;
      // }
    }
    // for (int dx = -1; dx <= 1; dx++) {
    //   for (int dy = -1; dy <= 1; dy++) {
    //     if (dx == 0 && dy == 0) {
    //       continue;
    //     }
    //     Eigen::Vector2i p(max_dis_x_index_list[i], max_dis_y_index_list[i]);
    //     if (0.8 * dis_array[p[0]][p[1]] < dis_array[p[0] + dx][p[1] + dy]) {
    //       is_add = true;
    //     }
    //   }
    // }
    if (is_add) {
      double px =
          mean_x_list[max_dis_x_index_list[i]][max_dis_y_index_list[i]] /
          img_count[max_dis_x_index_list[i]][max_dis_y_index_list[i]];
      double py =
          mean_y_list[max_dis_x_index_list[i]][max_dis_y_index_list[i]] /
          img_count[max_dis_x_index_list[i]][max_dis_y_index_list[i]];
      // std::cout << "max dis: " << max_dis_list[i] << std::endl;
      // std::cout << "px,py " << px << "," << py << std::endl;
      Eigen::Vector3d coord = py * x_axis + px * y_axis + projection_center;
      pcl::PointXYZ pi;
      pi.x = coord[0];
      pi.y = coord[1];
      pi.z = coord[2];
      key_point_list.push_back(coord);
      key_count_list.push_back(max_dis_list[i]);
      // for (size_t j = 0;
      //      j <
      //      edge_cloud_array[max_dis_x_index_list[i]][max_dis_y_index_list[i]]
      //          .size();
      //      j++) {
      //   edge_cloud->points.push_back(
      //       edge_cloud_array[max_dis_x_index_list[i]][max_dis_y_index_list[i]]
      //           .points[j]);
      // }

      // key_points->push_back(pi);
      // count_list.push_back(max_gradient_list[i]);
    }
  }
  std::vector<bool> is_add_list;
  for (size_t i = 0; i < key_point_list.size(); i++) {
    is_add_list.push_back(true);
  }
  if (max_constraint) {
    for (size_t i = 0; i < key_point_list.size(); i++) {
      Eigen::Vector3d pi = key_point_list[i];
      for (size_t j = 0; j < key_point_list.size(); j++) {
        Eigen::Vector3d pj = key_point_list[j];
        if (i != j) {
          double dis = sqrt(pow(pi[0] - pj[0], 2) + pow(pi[1] - pj[1], 2) +
                            pow(pi[2] - pj[2], 2));
          if (dis < 1) {
            if (key_count_list[i] > key_count_list[j]) {
              is_add_list[j] = false;
            } else {
              is_add_list[i] = false;
            }
          }
        }
      }
    }
    for (size_t i = 0; i < key_point_list.size(); i++) {
      if (is_add_list[i]) {
        pcl::PointXYZINormal pi;
        pi.x = key_point_list[i][0];
        pi.y = key_point_list[i][1];
        pi.z = key_point_list[i][2];
        pi.normal_x = projection_normal[0];
        pi.normal_y = projection_normal[1];
        pi.normal_z = projection_normal[2];
        key_points->points.push_back(pi);
        count_list.push_back(key_count_list[i]);
      }
    }
  } else {
    for (size_t i = 0; i < key_point_list.size(); i++) {
      pcl::PointXYZINormal pi;
      pi.x = key_point_list[i][0];
      pi.y = key_point_list[i][1];
      pi.z = key_point_list[i][2];
      pi.normal_x = projection_normal[0];
      pi.normal_y = projection_normal[1];
      pi.normal_z = projection_normal[2];
      key_points->points.push_back(pi);
      count_list.push_back(key_count_list[i]);
    }
  }
}

void binary_extractor(const std::vector<Plane *> &proj_plane_list,
                      const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
                      double max_count_threshold, double max_constraint_dis,
                      std::vector<BinaryDescriptor> &binary_descriptor_list) {
  bool max_constraint = true;
  std::vector<BinaryDescriptor> prepare_binary_list;
  std::vector<bool> is_add_list;
  Eigen::Vector3d project_normal = proj_plane_list[0]->normal;
  Eigen::Vector3d project_center = proj_plane_list[0]->center;
  pcl::PointCloud<pcl::PointXYZ>::Ptr edge_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr project_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);

  // std::cout << "here1" << std::endl;
  extract_binary(project_center, project_normal, max_count_threshold,
                 input_cloud, false, prepare_binary_list);
  // std::cout << "here2" << std::endl;

  pcl::PointCloud<pcl::PointXYZ>::Ptr prepare_key_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  for (size_t i = 0; i < prepare_binary_list.size(); i++) {
    is_add_list.push_back(true);
    pcl::PointXYZ pi;
    pi.x = prepare_binary_list[i].location[0];
    pi.y = prepare_binary_list[i].location[1];
    pi.z = prepare_binary_list[i].location[2];
    prepare_key_cloud->push_back(pi);
  }
  if (max_constraint) {
    pcl::KdTreeFLANN<pcl::PointXYZ> kd_tree;
    kd_tree.setInputCloud(prepare_key_cloud);
    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;
    float radius = max_constraint_dis;
    for (size_t i = 0; i < prepare_key_cloud->size(); i++) {
      pcl::PointXYZ searchPoint = prepare_key_cloud->points[i];
      if (kd_tree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch,
                               pointRadiusSquaredDistance) > 0) {
        Eigen::Vector3d pi(searchPoint.x, searchPoint.y, searchPoint.z);
        for (size_t j = 0; j < pointIdxRadiusSearch.size(); ++j) {
          if (pointIdxRadiusSearch[j] == i) {
            continue;
          }
          if (prepare_binary_list[i].summary <=
              prepare_binary_list[pointIdxRadiusSearch[j]].summary) {
            is_add_list[i] = false;
            // std::cout << "reject" << std::endl;
          }
        }
      }
    }
    for (size_t i = 0; i < is_add_list.size(); i++) {
      if (is_add_list[i]) {
        binary_descriptor_list.push_back(prepare_binary_list[i]);
      }
    }
  } else {
    for (size_t i = 0; i < prepare_binary_list.size(); i++) {
      binary_descriptor_list.push_back(prepare_binary_list[i]);
    }
  }
}

void binary_extractor(const Plane &proj_plane,
                      const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
                      const double max_count_threshold,
                      const double max_constraint_dis, const bool is_debug,
                      std::vector<BinaryDescriptor> &binary_descriptor_list) {
  bool max_constraint = (!is_debug);
  std::vector<BinaryDescriptor> prepare_binary_list;
  std::vector<bool> is_add_list;
  Eigen::Vector3d project_normal = proj_plane.normal;
  Eigen::Vector3d project_center = proj_plane.center;
  pcl::PointCloud<pcl::PointXYZ>::Ptr edge_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr project_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  extract_binary(project_center, project_normal, max_count_threshold,
                 input_cloud, is_debug, prepare_binary_list);
  pcl::PointCloud<pcl::PointXYZ>::Ptr prepare_key_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  for (size_t i = 0; i < prepare_binary_list.size(); i++) {
    is_add_list.push_back(true);
    pcl::PointXYZ pi;
    pi.x = prepare_binary_list[i].location[0];
    pi.y = prepare_binary_list[i].location[1];
    pi.z = prepare_binary_list[i].location[2];
    prepare_key_cloud->push_back(pi);
  }
  if (!is_debug) {
    pcl::KdTreeFLANN<pcl::PointXYZ> kd_tree;
    kd_tree.setInputCloud(prepare_key_cloud);
    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;
    float radius = max_constraint_dis;
    for (size_t i = 0; i < prepare_key_cloud->size(); i++) {
      pcl::PointXYZ searchPoint = prepare_key_cloud->points[i];
      if (kd_tree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch,
                               pointRadiusSquaredDistance) > 0) {
        Eigen::Vector3d pi(searchPoint.x, searchPoint.y, searchPoint.z);
        for (size_t j = 0; j < pointIdxRadiusSearch.size(); ++j) {
          if (pointIdxRadiusSearch[j] == i) {
            continue;
          }
          if (prepare_binary_list[i].summary <=
              prepare_binary_list[pointIdxRadiusSearch[j]].summary) {
            is_add_list[i] = false;
            // std::cout << "reject" << std::endl;
          }
        }
      }
    }
    for (size_t i = 0; i < is_add_list.size(); i++) {
      if (is_add_list[i]) {
        binary_descriptor_list.push_back(prepare_binary_list[i]);
      }
    }
  } else {
    for (size_t i = 0; i < prepare_binary_list.size(); i++) {
      binary_descriptor_list.push_back(prepare_binary_list[i]);
    }
  }
}

void corner_extractor(std::vector<Plane *> &proj_plane_list,
                      pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
                      double max_count_threshold, double max_constraint_dis,
                      pcl::PointCloud<pcl::PointXYZINormal>::Ptr &key_cloud,
                      std::vector<double> &count_list) {
  bool max_constraint = true;
  pcl::PointCloud<pcl::PointXYZINormal>::Ptr prepare_key_cloud(
      new pcl::PointCloud<pcl::PointXYZINormal>);
  std::vector<double> pre_count_list;
  std::vector<bool> is_add_list;
  Eigen::Vector3d project_normal = proj_plane_list[0]->normal;
  Eigen::Vector3d project_center = proj_plane_list[0]->center;
  pcl::PointCloud<pcl::PointXYZ>::Ptr edge_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr project_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  std::cout << "normal:" << project_normal.transpose()
            << ", center:" << project_center.transpose()
            << ", size:" << input_cloud->size() << std::endl;
  extract_corner4(project_center, project_normal, max_constraint_dis,
                  input_cloud, edge_cloud, project_cloud, prepare_key_cloud,
                  pre_count_list);
  for (size_t i = 0; i < pre_count_list.size(); i++) {
    is_add_list.push_back(true);
  }
  if (max_constraint) {
    pcl::KdTreeFLANN<pcl::PointXYZINormal> kd_tree;
    kd_tree.setInputCloud(prepare_key_cloud);
    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;
    float radius = max_constraint_dis;
    for (size_t i = 0; i < prepare_key_cloud->size(); i++) {
      pcl::PointXYZINormal searchPoint = prepare_key_cloud->points[i];
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
            // std::cout << "reject" << std::endl;
          }
        }
      }
    }
    for (size_t i = 0; i < is_add_list.size(); i++) {
      if (is_add_list[i]) {
        key_cloud->points.push_back(prepare_key_cloud->points[i]);
        count_list.push_back(pre_count_list[i]);
      }
    }
  } else {
    for (size_t i = 0; i < prepare_key_cloud->size(); i++) {
      key_cloud->points.push_back(prepare_key_cloud->points[i]);
      count_list.push_back(pre_count_list[i]);
    }
  }
}

void corner_extractor_debug(
    std::unordered_map<VOXEL_LOC, OctoTree *> &feat_map,
    double max_count_threshold, double max_constraint_dis,
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr &plane_cloud,
    pcl::PointCloud<pcl::PointXYZ>::Ptr &edge_cloud,
    pcl::PointCloud<pcl::PointXYZ>::Ptr &project_cloud,
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr &key_cloud,
    std::vector<double> &count_list, std::vector<Eigen::Vector3d> &rgb_list) {
  srand((unsigned)std::time(NULL));
  pcl::PointCloud<pcl::PointXYZINormal>::Ptr prepare_key_cloud(
      new pcl::PointCloud<pcl::PointXYZINormal>);
  std::vector<double> pre_count_list;
  bool max_constraint = true;
  std::vector<bool> is_add_list;
  std::vector<Eigen::Vector3i> voxel_round;
  for (int x = -1; x <= 1; x++) {
    for (int y = -1; y <= 1; y++) {
      for (int z = -1; z <= 1; z++) {
        Eigen::Vector3i voxel_inc(x, y, z);
        voxel_round.push_back(voxel_inc);
      }
    }
  }
  OctoTree *current_octo = nullptr;
  for (auto iter = feat_map.begin(); iter != feat_map.end(); iter++) {
    if (iter->second->plane_ptr_->is_plane) {
      current_octo = iter->second;
      int connect_num = 0;
      for (int i = 0; i < 6; i++) {
        if (current_octo->connect_[i]) {
          connect_num++;
        }
      }
      Eigen::Vector3d plane_rgb(0, 0, 0);
      plane_rgb << iter->second->plane_ptr_->normal;

      if (connect_num >= 1) {
        for (size_t i = 0; i < current_octo->temp_cloud_->size(); i++) {
          pcl::PointXYZRGB prgb;
          prgb.x = current_octo->temp_cloud_->points[i].x;
          prgb.y = current_octo->temp_cloud_->points[i].y;
          prgb.z = current_octo->temp_cloud_->points[i].z;
          prgb.r = fabs(plane_rgb[0]) * 255.0;
          prgb.g = fabs(plane_rgb[1]) * 255.0;
          prgb.b = fabs(plane_rgb[2]) * 255.0;
          plane_cloud->points.push_back(prgb);
        }
      }
    } else {
      Eigen::Vector3d last_projection_normal(0, 0, 0);
      VOXEL_LOC current_position = iter->first;
      current_octo = iter->second;
      int connect_index = -1;
      for (int i = 0; i < 6; i++) {
        if (current_octo->connect_[i]) {
          connect_index = i;
          OctoTree *connect_octo = current_octo->connect_tree_[connect_index];
          bool use = false;
          for (int j = 0; j < 6; j++) {
            if (connect_octo->is_check_connect_[j]) {
              if (connect_octo->connect_[j]) {
                use = true;
              }
            }
          }
          if (use == false) {
            continue;
          }
          if (current_octo->temp_cloud_->size() > 10) {
            Eigen::Vector3d near_center =
                current_octo->connect_tree_[connect_index]->plane_ptr_->center;
            Eigen::Vector3d projection_normal =
                current_octo->connect_tree_[connect_index]->plane_ptr_->normal;
            Eigen::Vector3d projection_center = near_center;
            pcl::PointCloud<pcl::PointXYZI>::Ptr input_cloud(
                new pcl::PointCloud<pcl::PointXYZI>);
            for (auto voxel_inc : voxel_round) {
              VOXEL_LOC connect_project_position = current_position;
              connect_project_position.x += voxel_inc[0];
              connect_project_position.y += voxel_inc[1];
              connect_project_position.z += voxel_inc[2];
              auto iter_near = feat_map.find(connect_project_position);
              if (iter_near != feat_map.end()) {
                bool skip_flag = false;
                if (!feat_map[connect_project_position]->plane_ptr_->is_plane) {
                  if (feat_map[connect_project_position]->is_project_) {
                    for (auto normal :
                         feat_map[connect_project_position]->project_normal) {
                      Eigen::Vector3d normal_diff = projection_normal - normal;
                      Eigen::Vector3d normal_add = projection_normal + normal;
                      if (normal_diff.norm() < 0.5 || normal_add.norm() < 0.5) {
                        skip_flag = true;
                      }
                    }
                  }
                  if (skip_flag) {
                    // std::cout << "skip muti projection" << std::endl;
                    continue;
                  }
                  for (size_t j = 0;
                       j <
                       feat_map[connect_project_position]->temp_cloud_->size();
                       j++) {
                    input_cloud->points.push_back(
                        feat_map[connect_project_position]
                            ->temp_cloud_->points[j]);
                    feat_map[connect_project_position]->is_project_ = true;
                    feat_map[connect_project_position]
                        ->project_normal.push_back(projection_normal);
                  }
                }
              }
            }
            pcl::PointCloud<pcl::PointXYZ>::Ptr sub_edge_cloud(
                new pcl::PointCloud<pcl::PointXYZ>);
            pcl::PointCloud<pcl::PointXYZ>::Ptr sub_project_cloud(
                new pcl::PointCloud<pcl::PointXYZ>);
            pcl::PointCloud<pcl::PointXYZINormal>::Ptr sub_key_cloud(
                new pcl::PointCloud<pcl::PointXYZINormal>);
            std::vector<double> sub_count_list;
            // if (fabs(projection_normal[2]) > 0.8)
            // projection_normal << 0, 0, 1;
            extract_corner3(projection_center, projection_normal,
                            max_count_threshold, input_cloud, sub_edge_cloud,
                            sub_project_cloud, sub_key_cloud, sub_count_list);
            //
            int N = 999;
            float r = rand() % (N + 1) / (float)(N + 1);
            float g = rand() % (N + 1) / (float)(N + 1);
            float b = rand() % (N + 1) / (float)(N + 1);
            Eigen::Vector3d single_rgb(r, g, b);
            for (size_t i = 0; i < sub_key_cloud->size(); i++) {
              prepare_key_cloud->points.push_back(sub_key_cloud->points[i]);
              pre_count_list.push_back(sub_count_list[i]);
              is_add_list.push_back(true);
              rgb_list.push_back(single_rgb);
            }
            for (size_t i = 0; i < sub_project_cloud->size(); i++) {
              project_cloud->points.push_back(sub_project_cloud->points[i]);
            }
            for (size_t i = 0; i < sub_edge_cloud->size(); i++) {
              edge_cloud->points.push_back(sub_edge_cloud->points[i]);
            }
          }
        }
      }
    }
  }
  if (max_constraint) {
    pcl::KdTreeFLANN<pcl::PointXYZINormal> kd_tree;
    kd_tree.setInputCloud(prepare_key_cloud);
    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;
    float radius = max_constraint_dis;
    for (size_t i = 0; i < prepare_key_cloud->size(); i++) {
      pcl::PointXYZINormal searchPoint = prepare_key_cloud->points[i];
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
            // std::cout << "reject" << std::endl;
          }
        }
      }
    }
    for (size_t i = 0; i < is_add_list.size(); i++) {
      if (is_add_list[i]) {
        key_cloud->points.push_back(prepare_key_cloud->points[i]);
        count_list.push_back(pre_count_list[i]);
      }
    }
  } else {
    for (size_t i = 0; i < prepare_key_cloud->size(); i++) {
      key_cloud->points.push_back(prepare_key_cloud->points[i]);
      count_list.push_back(pre_count_list[i]);
    }
  }
}

// void add_norepeat_descriptor(
//     std::unordered_map<DESCRIPTOR_LOC, std::vector<Descriptor>> &feat_map,
//     std::vector<Descriptor> &current_descriptor) {
//   for (auto descrip : current_descriptor) {
//     DESCRIPTOR_LOC position;
//     position.x = (int)(descrip.triangle[0] + 0.5);
//     position.y = (int)(descrip.triangle[1] + 0.5);
//     position.z = (int)(descrip.triangle[2] + 0.5);
//     position.a = (int)(descrip.angle[0]);
//     position.b = (int)(descrip.angle[1]);
//     position.c = (int)(descrip.angle[2]);
//     auto iter = feat_map.find(position);
//     if (iter != feat_map.end()) {
//       for(size_t i=0;i<feat_map[position].size();i++){

//       }
//       feat_map[position].push_back(descrip);
//     } else {
//       std::vector<Descriptor> descriptor_list;
//       descriptor_list.push_back(descrip);
//       feat_map[position] = descriptor_list;
//     }
//   }
// }

void add_descriptor(
    std::unordered_map<DESCRIPTOR_LOC, std::vector<Descriptor>> &feat_map,
    std::vector<Descriptor> &current_descriptor) {
  for (auto descrip : current_descriptor) {
    DESCRIPTOR_LOC position;
    position.x = (int)(descrip.triangle[0] + 0.5);
    position.y = (int)(descrip.triangle[1] + 0.5);
    position.z = (int)(descrip.triangle[2] + 0.5);
    position.a = (int)(descrip.angle[0]);
    position.b = (int)(descrip.angle[1]);
    position.c = (int)(descrip.angle[2]);
    auto iter = feat_map.find(position);
    descrip.score_frame.push_back(descrip.frame_number);
    if (iter != feat_map.end()) {
      feat_map[position].push_back(descrip);
    } else {
      std::vector<Descriptor> descriptor_list;
      descriptor_list.push_back(descrip);
      feat_map[position] = descriptor_list;
    }
  }
}

void add_descriptor_new(
    std::unordered_map<DESCRIPTOR_LOC, std::vector<Descriptor>> &feat_map,
    std::vector<Descriptor> &current_descriptor) {
  int add_cnt = 0;
  int new_cnt = 0;
  for (auto descrip : current_descriptor) {
    DESCRIPTOR_LOC position;
    position.x = (int)(descrip.triangle[0] + 0.5);
    position.y = (int)(descrip.triangle[1] + 0.5);
    position.z = (int)(descrip.triangle[2] + 0.5);
    // position.a = (int)(descrip.angle[0]);
    // position.b = (int)(descrip.angle[1]);
    // position.c = (int)(descrip.angle[2]);
    auto iter = feat_map.find(position);
    if (iter != feat_map.end()) {
      double dis_threshold = descrip.triangle.norm() * 0.005;
      bool is_add = true;
      for (size_t i = 0; i < feat_map[position].size(); i++) {
        Descriptor map_descriptor = feat_map[position][i];
        double similarity =
            (binary_similarity(descrip.binary_A, map_descriptor.binary_A) +
             binary_similarity(descrip.binary_B, map_descriptor.binary_B) +
             binary_similarity(descrip.binary_C, map_descriptor.binary_C)) /
            3;
        if (v3d_dis(descrip.triangle, map_descriptor.triangle) <
                dis_threshold &&
            similarity > 0.75) {
          is_add = false;
          feat_map[position][i].score_frame.push_back(descrip.frame_number);
          feat_map[position][i].position_list.push_back(
              descrip.position_list[0]);
          break;
        }
      }
      if (is_add) {
        add_cnt++;
        feat_map[position].push_back(descrip);
      }
    } else {
      std::vector<Descriptor> descriptor_list;
      descriptor_list.push_back(descrip);
      feat_map[position] = descriptor_list;
      add_cnt++;
      new_cnt++;
    }
  }
  std::cout << "all size:" << current_descriptor.size()
            << ", add size:" << add_cnt << ", new size:" << new_cnt
            << std::endl;
}

void rough_loop_detection(
    std::unordered_map<DESCRIPTOR_LOC, std::vector<Descriptor>> &feat_map,
    std::vector<Descriptor> &current_descriptor, const int alternative_num,
    std::vector<MatchTriangleList> &alternative_match) {
  int outlier = 0;
  double max_dis = 100;
  double match_array[10000] = {0};
  std::vector<std::pair<Descriptor, Descriptor>> match_list;
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
  std::vector<bool> useful_match(current_descriptor.size());
  std::vector<std::vector<size_t>> useful_match_index(
      current_descriptor.size());
  std::vector<std::vector<DESCRIPTOR_LOC>> useful_match_position(
      current_descriptor.size());
  std::vector<size_t> index(current_descriptor.size());
  for (size_t i = 0; i < index.size(); ++i) {
    index[i] = i;
    useful_match[i] = false;
  }
  std::mutex mylock;
  auto t0 = std::chrono::high_resolution_clock::now();
  std::for_each(
      std::execution::par_unseq, index.begin(), index.end(),
      [&](const size_t &i) {
        Descriptor descrip = current_descriptor[i];
        DESCRIPTOR_LOC position;
        int best_index = 0;
        DESCRIPTOR_LOC best_position;
        double dis_threshold = descrip.triangle.norm() * 0.01; // old 0.005
        for (auto voxel_inc : voxel_round) {
          position.x = (int)(descrip.triangle[0] + voxel_inc[0]);
          position.y = (int)(descrip.triangle[1] + voxel_inc[1]);
          position.z = (int)(descrip.triangle[2] + voxel_inc[2]);
          Eigen::Vector3d voxel_center((double)position.x + 0.5,
                                       (double)position.y + 0.5,
                                       (double)position.z + 0.5);
          if (v3d_dis(descrip.triangle, voxel_center) < 1.5) {
            auto iter = feat_map.find(position);
            if (iter != feat_map.end()) {
              for (size_t j = 0; j < feat_map[position].size(); j++) {
                if ((descrip.center - feat_map[position][j].center).norm() <
                    max_dis)
                  if ((descrip.frame_number -
                       feat_map[position][j].frame_number) > 25) {
                    double dis = v3d_dis(descrip.triangle,
                                         feat_map[position][j].triangle);
                    if (dis < dis_threshold) {
                      mylock.lock();
                      useful_match[i] = true;
                      useful_match_position[i].push_back(position);
                      useful_match_index[i].push_back(j);
                      mylock.unlock();
                    }
                  }
              }
            }
          }
        }
      });
  auto t1 = std::chrono::high_resolution_clock::now();
  std::for_each(
      std::execution::par_unseq, index.begin(), index.end(),
      [&](const size_t &i) {
        if (useful_match[i]) {
          std::pair<Descriptor, Descriptor> single_match_pair;
          single_match_pair.first = current_descriptor[i];
          for (size_t j = 0; j < useful_match_index[i].size(); j++) {
            single_match_pair.second =
                feat_map[useful_match_position[i][j]][useful_match_index[i][j]];
            double similarity =
                (binary_similarity(single_match_pair.first.binary_A,
                                   single_match_pair.second.binary_A) +
                 binary_similarity(single_match_pair.first.binary_B,
                                   single_match_pair.second.binary_B) +
                 binary_similarity(single_match_pair.first.binary_C,
                                   single_match_pair.second.binary_C)) /
                3;
            // old 0.75
            // double similarity = 1;
            similarity = 1;
            if (similarity > 0.75) {
              mylock.lock();
              match_array[single_match_pair.second.frame_number] += similarity;
              match_list.push_back(single_match_pair);
              match_list_index.push_back(single_match_pair.second.frame_number);
              mylock.unlock();
            }
          }
        }
      });
  // int cnt = 0;
  // for (size_t i = 0; i < useful_match.size(); i++) {
  //   if (useful_match[i]) {
  //     std::pair<Descriptor, Descriptor> single_match_pair;
  //     single_match_pair.first = current_descriptor[i];
  //     for (size_t j = 0; j < useful_match_index[i].size(); j++) {
  //       cnt++;
  //       single_match_pair.second =
  //           feat_map[useful_match_position[i][j]][useful_match_index[i][j]];
  //       match_array[single_match_pair.second.frame_number] += 1;
  //       match_list.push_back(single_match_pair);
  //       match_list_index.push_back(single_match_pair.second.frame_number);
  //     }
  //   }
  // }
  // std::cout << "useful cnt: " << cnt << std::endl;
  auto t2 = std::chrono::high_resolution_clock::now();
  for (int cnt = 0; cnt < alternative_num; cnt++) {
    double max_vote = 1;
    int max_vote_index = -1;
    for (int i = 0; i < 5000; i++) {
      if (match_array[i] > max_vote) {
        max_vote = match_array[i];
        max_vote_index = i;
      }
    }
    MatchTriangleList match_triangle_list;
    if (max_vote_index >= 0) {
      match_array[max_vote_index] = 0;
      match_triangle_list.match_frame = max_vote_index;
      double mean_dis = 0;
      for (size_t i = 0; i < match_list_index.size(); i++) {
        if (match_list_index[i] == max_vote_index) {
          match_triangle_list.match_list.push_back(match_list[i]);
        }
      }
      // for (auto single_match_pair : match_list) {
      //   if (single_match_pair.second.frame_number == max_vote_index) {
      //     match_triangle_list.match_list.push_back(single_match_pair);
      //     // double dis = 2 *
      //     //              v3d_dis(single_match_pair.first.triangle,
      //     //                      single_match_pair.second.triangle) /
      //     //              (single_match_pair.first.triangle.norm() +
      //     //               single_match_pair.second.triangle.norm());
      //     // mean_dis += dis;
      //   }
      // }
      // match_triangle_list.mean_dis =
      //     mean_dis / match_triangle_list.match_list.size();
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

void rough_loop_detect(
    std::unordered_map<DESCRIPTOR_LOC, std::vector<Descriptor>> &feat_map,
    std::vector<Descriptor> &current_descriptor, const int alternative_num,
    std::vector<MatchTriangleList> &alternative_match) {
  int outlier = 0;
  double max_dis = 100;
  double match_array[10000] = {0};
  std::vector<std::pair<Descriptor, Descriptor>> match_list;
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
  std::vector<size_t> index(current_descriptor.size());
  for (size_t i = 0; i < index.size(); ++i) {
    index[i] = i;
  }
  auto t0 = std::chrono::high_resolution_clock::now();
  std::vector<bool> useful_match(current_descriptor.size());
  std::vector<size_t> useful_match_index(current_descriptor.size());
  std::vector<DESCRIPTOR_LOC> useful_match_position(current_descriptor.size());
  std::for_each(std::execution::par_unseq, index.begin(), index.end(),
                [&](const size_t &i) {
                  Descriptor descrip = current_descriptor[i];
                  DESCRIPTOR_LOC position;
                  double best_dis = 100;
                  int best_index = 0;
                  DESCRIPTOR_LOC best_position;
                  for (auto voxel_inc : voxel_round) {
                    position.x = (int)(descrip.triangle[0] + voxel_inc[0]);
                    position.y = (int)(descrip.triangle[1] + voxel_inc[1]);
                    position.z = (int)(descrip.triangle[2] + voxel_inc[2]);
                    Eigen::Vector3d voxel_center((double)position.x + 0.5,
                                                 (double)position.y + 0.5,
                                                 (double)position.z + 0.5);
                    if (v3d_dis(descrip.triangle, voxel_center) < 1.5) {
                      auto iter = feat_map.find(position);
                      if (iter != feat_map.end()) {
                        for (size_t j = 0; j < feat_map[position].size(); j++) {
                          if ((descrip.frame_number -
                               feat_map[position][j].frame_number) > 20) {
                            double dis = v3d_dis(
                                descrip.triangle / descrip.triangle_scale,
                                feat_map[position][j].triangle /
                                    feat_map[position][j].triangle_scale);
                            if (dis < best_dis) {
                              best_dis = dis;
                              best_index = j;
                              best_position = position;
                            }
                          }
                        }
                      }
                    }
                  }
                  double dis_threshold =
                      descrip.triangle.norm() / descrip.triangle_scale * 0.01;
                  if (best_dis < dis_threshold) {
                    useful_match[i] = true;
                    useful_match_index[i] = best_index;
                    useful_match_position[i] = best_position;
                  } else {
                    useful_match[i] = false;
                  }
                });
  auto t1 = std::chrono::high_resolution_clock::now();
  std::mutex mylock;
  std::for_each(
      std::execution::par_unseq, index.begin(), index.end(),
      [&](const size_t &i) {
        if (useful_match[i]) {
          std::pair<Descriptor, Descriptor> single_match_pair;
          single_match_pair.first = current_descriptor[i];
          single_match_pair.second =
              feat_map[useful_match_position[i]][useful_match_index[i]];
          double similarity =
              (binary_similarity(single_match_pair.first.binary_A,
                                 single_match_pair.second.binary_A) +
               binary_similarity(single_match_pair.first.binary_B,
                                 single_match_pair.second.binary_B) +
               binary_similarity(single_match_pair.first.binary_C,
                                 single_match_pair.second.binary_C)) /
              3;
          if (similarity > 0.75) {
            mylock.lock();
            match_array[single_match_pair.second.frame_number] += similarity;
            match_list.push_back(single_match_pair);
            match_list_index.push_back(single_match_pair.second.frame_number);
            mylock.unlock();
          }
        }
      });

  // for (size_t i = 0; i < useful_match.size(); i++) {
  //   if (useful_match[i]) {
  //     std::pair<Descriptor, Descriptor> single_match_pair;
  //     single_match_pair.first = current_descriptor[i];
  //     single_match_pair.second =
  //         feat_map[useful_match_position[i]][useful_match_index[i]];
  //     double similarity =
  //         (binary_similarity(single_match_pair.first.binary_A,
  //                            single_match_pair.second.binary_A) +
  //          binary_similarity(single_match_pair.first.binary_B,
  //                            single_match_pair.second.binary_B) +
  //          binary_similarity(single_match_pair.first.binary_C,
  //                            single_match_pair.second.binary_C)) /
  //         3;
  //     if (similarity > 0.75) {
  //       match_array[single_match_pair.second.frame_number] += similarity;
  //       match_list.push_back(single_match_pair);
  //     }
  //   }
  // }
  auto t2 = std::chrono::high_resolution_clock::now();

  for (int cnt = 0; cnt < alternative_num; cnt++) {
    double max_vote = 1;
    int max_vote_index = -1;
    for (int i = 0; i < 5000; i++) {
      if (match_array[i] > max_vote) {
        max_vote = match_array[i];
        max_vote_index = i;
      }
    }
    MatchTriangleList match_triangle_list;
    if (max_vote_index >= 0) {
      match_array[max_vote_index] = 0;
      match_triangle_list.match_frame = max_vote_index;
      double mean_dis = 0;
      for (size_t i = 0; i < match_list_index.size(); i++) {
        if (match_list_index[i] == max_vote_index) {
          match_triangle_list.match_list.push_back(match_list[i]);
        }
      }
      // for (auto single_match_pair : match_list) {
      //   if (single_match_pair.second.frame_number == max_vote_index) {
      //     match_triangle_list.match_list.push_back(single_match_pair);
      //     // double dis = 2 *
      //     //              v3d_dis(single_match_pair.first.triangle,
      //     //                      single_match_pair.second.triangle) /
      //     //              (single_match_pair.first.triangle.norm() +
      //     //               single_match_pair.second.triangle.norm());
      //     // mean_dis += dis;
      //   }
      // }
      // match_triangle_list.mean_dis =
      //     mean_dis / match_triangle_list.match_list.size();
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

void rough_loop_detect_new(
    std::unordered_map<DESCRIPTOR_LOC, std::vector<Descriptor>> &feat_map,
    std::vector<Descriptor> &current_descriptor, const int alternative_num,
    std::vector<MatchTriangleList> &alternative_match) {
  int outlier = 0;
  double max_dis = 100;
  double match_array[10000] = {0};
  std::vector<std::pair<Descriptor, Descriptor>> match_list;
  std::vector<int> match_descriptor_vec;
  std::vector<DESCRIPTOR_LOC> match_voxel_loc_vec;
  std::vector<int> match_vect_index_vec;
  std::vector<Eigen::Vector3i> voxel_round;
  for (int x = -1; x <= 1; x++) {
    for (int y = -1; y <= 1; y++) {
      for (int z = -1; z <= 1; z++) {
        Eigen::Vector3i voxel_inc(x, y, z);
        voxel_round.push_back(voxel_inc);
      }
    }
  }
  std::vector<bool> useful_match(current_descriptor.size());
  std::vector<std::vector<size_t>> useful_match_index(
      current_descriptor.size());
  std::vector<std::vector<DESCRIPTOR_LOC>> useful_match_position(
      current_descriptor.size());
  std::vector<size_t> index(current_descriptor.size());
  for (size_t i = 0; i < index.size(); ++i) {
    index[i] = i;
    useful_match[i] = false;
  }
  std::mutex mylock;
  auto t0 = std::chrono::high_resolution_clock::now();
  std::for_each(
      std::execution::par_unseq, index.begin(), index.end(),
      [&](const size_t &i) {
        Descriptor descrip = current_descriptor[i];
        DESCRIPTOR_LOC position;
        int best_index = 0;
        DESCRIPTOR_LOC best_position;
        double dis_threshold = descrip.triangle.norm() * 0.005;
        for (auto voxel_inc : voxel_round) {
          position.x = (int)(descrip.triangle[0] + voxel_inc[0]);
          position.y = (int)(descrip.triangle[1] + voxel_inc[1]);
          position.z = (int)(descrip.triangle[2] + voxel_inc[2]);
          Eigen::Vector3d voxel_center((double)position.x + 0.5,
                                       (double)position.y + 0.5,
                                       (double)position.z + 0.5);
          if (v3d_dis(descrip.triangle, voxel_center) < 1.5) {
            auto iter = feat_map.find(position);
            if (iter != feat_map.end()) {
              for (size_t j = 0; j < feat_map[position].size(); j++) {
                if ((descrip.frame_number -
                     feat_map[position][j].frame_number) > 20) {
                  double dis =
                      v3d_dis(descrip.triangle, feat_map[position][j].triangle);
                  if (dis < dis_threshold) {
                    double similarity =
                        (binary_similarity(descrip.binary_A,
                                           feat_map[position][j].binary_A) +
                         binary_similarity(descrip.binary_B,
                                           feat_map[position][j].binary_B) +
                         binary_similarity(descrip.binary_C,
                                           feat_map[position][j].binary_C)) /
                        3;
                    if (similarity >= 0.8) {
                      mylock.lock();
                      useful_match[i] = true;
                      match_descriptor_vec.push_back(i);
                      match_voxel_loc_vec.push_back(position);
                      match_vect_index_vec.push_back(j);
                      mylock.unlock();
                    }
                  }
                }
              }
            }
          }
        }
      });
  auto t1 = std::chrono::high_resolution_clock::now();

  std::vector<size_t> index2;
  for (size_t i = 0; i < match_descriptor_vec.size(); i++) {
    index2.push_back(i);
  }
  std::for_each(
      std::execution::par_unseq, index2.begin(), index2.end(),
      [&](const size_t &i) {
        // mylock.lock();
        // match_array[feat_map[match_voxel_loc_vec[i]][match_vect_index_vec[i]]
        //                 .frame_number] += 1;
        // mylock.unlock();
        for (size_t k = 0;
             k < feat_map[match_voxel_loc_vec[i]][match_vect_index_vec[i]]
                     .score_frame.size();
             k++) {
          mylock.lock();
          match_array[feat_map[match_voxel_loc_vec[i]][match_vect_index_vec[i]]
                          .score_frame[k]] += 1;
          mylock.unlock();
        }
      });
  auto t2 = std::chrono::high_resolution_clock::now();
  for (int cnt = 0; cnt < alternative_num; cnt++) {
    double max_vote = 1;
    int max_vote_index = -1;
    for (int i = 0; i < 5000; i++) {
      if (match_array[i] > max_vote) {
        max_vote = match_array[i];
        max_vote_index = i;
      }
    }
    MatchTriangleList match_triangle_list;
    if (max_vote_index >= 0) {
      match_array[max_vote_index] = 0;
      match_triangle_list.match_frame = max_vote_index;
      std::for_each(
          std::execution::par_unseq, index2.begin(), index2.end(),
          [&](const size_t &i) {
            // if
            // (feat_map[match_voxel_loc_vec[i]][match_vect_index_vec[i]]
            //         .frame_number == max_vote_index) {
            //   std::pair<Descriptor, Descriptor> single_match_pair;
            //   single_match_pair.first =
            //       current_descriptor[match_descriptor_vec[i]];
            //   single_match_pair.second =
            //       feat_map[match_voxel_loc_vec[i]][match_vect_index_vec[i]];
            //   mylock.lock();
            //   match_triangle_list.match_list.push_back(single_match_pair);
            //   mylock.unlock();
            // }
            std::pair<Descriptor, Descriptor> single_match_pair;
            single_match_pair.first =
                current_descriptor[match_descriptor_vec[i]];
            for (size_t k = 0;
                 k < feat_map[match_voxel_loc_vec[i]][match_vect_index_vec[i]]
                         .score_frame.size();
                 k++) {
              if (feat_map[match_voxel_loc_vec[i]][match_vect_index_vec[i]]
                      .score_frame[k] == max_vote_index) {
                single_match_pair.second =
                    feat_map[match_voxel_loc_vec[i]][match_vect_index_vec[i]];
                single_match_pair.second.A =
                    single_match_pair.second.position_list[k].block<3, 1>(0, 0);
                single_match_pair.second.B =
                    single_match_pair.second.position_list[k].block<3, 1>(0, 1);
                single_match_pair.second.C =
                    single_match_pair.second.position_list[k].block<3, 1>(0, 2);
                mylock.lock();
                match_triangle_list.match_list.push_back(single_match_pair);
                mylock.unlock();
              }
            }
          });
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

void triangle_solver(std::pair<Descriptor, Descriptor> &triangle_pair,
                     Eigen::Matrix3d &rot, Eigen::Vector3d &t) {
  Eigen::Matrix3d src = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d ref = Eigen::Matrix3d::Zero();
  src.col(0) = triangle_pair.first.A - triangle_pair.first.center;
  src.col(1) = triangle_pair.first.B - triangle_pair.first.center;
  src.col(2) = triangle_pair.first.C - triangle_pair.first.center;
  ref.col(0) = triangle_pair.second.A - triangle_pair.second.center;
  ref.col(1) = triangle_pair.second.B - triangle_pair.second.center;
  ref.col(2) = triangle_pair.second.C - triangle_pair.second.center;
  Eigen::Matrix3d covariance = src * ref.transpose();
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(covariance, Eigen::ComputeThinU |
                                                        Eigen::ComputeThinV);
  Eigen::Matrix3d V = svd.matrixV();
  Eigen::Matrix3d U = svd.matrixU();
  rot = V * U.transpose();
  if (rot.determinant() < 0) {
    Eigen::Matrix3d K;
    K << 1, 0, 0, 0, 1, 0, 0, 0, -1;
    rot = V * K * U.transpose();
  }
  t = -rot * triangle_pair.first.center + triangle_pair.second.center;
  // std::cout << "Rot:" << rot << std::endl;
  // std::cout << "T:" << t.transpose() << std::endl;
}
void fine_loop_detect(
    std::vector<std::pair<Descriptor, Descriptor>> &match_list, bool &is_sucess,
    Eigen::Matrix3d &rot, Eigen::Vector3d &t,
    std::vector<std::pair<Descriptor, Descriptor>> &sucess_match_list) {
  // old 0.25,0.5 and 0.1
  // velodyne 0.5

  double dis_threshold = 1;
  is_sucess = false;
  int max_vote = 0;
  std::vector<std::pair<Descriptor, Descriptor>> unsucess_match_list;
  std::time_t solve_time = 0;
  std::time_t verify_time = 0;
  for (size_t i = 0; i < match_list.size(); i = i + 1) {
    auto single_pair = match_list[i];
    int vote = 0;
    Eigen::Matrix3d test_rot;
    Eigen::Vector3d test_t;
    std::vector<std::pair<Descriptor, Descriptor>> temp_match_list;
    std::vector<std::pair<Descriptor, Descriptor>> temp_unmatch_list;
    std::time_t temp_t1 = clock();
    triangle_solver(single_pair, test_rot, test_t);
    std::time_t temp_t2 = clock();
    solve_time += (temp_t2 - temp_t1);

    for (auto verify_pair : match_list) {
      Eigen::Vector3d A = verify_pair.first.A;
      Eigen::Vector3d A_transform = test_rot * A + test_t;
      Eigen::Vector3d B = verify_pair.first.B;
      Eigen::Vector3d B_transform = test_rot * B + test_t;
      Eigen::Vector3d C = verify_pair.first.C;
      Eigen::Vector3d C_transform = test_rot * C + test_t;
      double dis_A = v3d_dis(A_transform, verify_pair.second.A);
      double dis_B = v3d_dis(B_transform, verify_pair.second.B);
      double dis_C = v3d_dis(C_transform, verify_pair.second.C);
      if (dis_A < dis_threshold && dis_B < dis_threshold &&
          dis_C < dis_threshold) {
        temp_match_list.push_back(verify_pair);
        vote++;
      } else {
        temp_unmatch_list.push_back(verify_pair);
      }
    }
    std::time_t temp_t3 = clock();
    verify_time += (temp_t3 - temp_t2);
    if (vote > max_vote) {
      max_vote = vote;
      rot = test_rot;
      t = test_t;
      sucess_match_list.clear();
      unsucess_match_list.clear();
      for (auto var : temp_match_list) {
        sucess_match_list.push_back(var);
      }
      for (auto var : temp_unmatch_list) {
        unsucess_match_list.push_back(var);
      }
    }
  }
  // std::cout << "time for triangle solve: "
  //           << (double)solve_time / CLOCKS_PER_SEC * 1000 << " ms" <<
  //           std::endl;
  // std::cout << "time for triangle verify: "
  //           << (double)verify_time / CLOCKS_PER_SEC * 1000 << " ms"
  //           << std::endl;
  if (max_vote >= 5) {
    is_sucess = true;
    // std::cout << "sucefully match, vote num:" << max_vote << std::endl;
    double mean_sucess_dis = 0;
    double mean_unsucess_dis = 0;
    for (auto var : sucess_match_list) {
      double min_similarity =
          std::min(binary_similarity(var.first.binary_A, var.second.binary_A),
                   binary_similarity(var.first.binary_B, var.second.binary_B));
      min_similarity =
          std::min(min_similarity,
                   binary_similarity(var.first.binary_C, var.second.binary_C));
      mean_sucess_dis += min_similarity;
    }
    for (auto var : unsucess_match_list) {
      double min_similarity =
          std::min(binary_similarity(var.first.binary_A, var.second.binary_A),
                   binary_similarity(var.first.binary_B, var.second.binary_B));
      min_similarity =
          std::min(min_similarity,
                   binary_similarity(var.first.binary_C, var.second.binary_C));
      mean_unsucess_dis += min_similarity;
    }
    std::cout << "fine mean similarity:"
              << mean_sucess_dis / sucess_match_list.size()
              << " , bad mean similarity:"
              << mean_unsucess_dis / unsucess_match_list.size() << std::endl;
  }
}

void fine_loop_detect_tbb(
    std::vector<std::pair<Descriptor, Descriptor>> &match_list, bool &is_sucess,
    Eigen::Matrix3d &rot, Eigen::Vector3d &t,
    std::vector<std::pair<Descriptor, Descriptor>> &sucess_match_list) {
  double dis_threshold = 3;
  is_sucess = false;
  std::vector<std::pair<Descriptor, Descriptor>> unsucess_match_list;
  std::time_t solve_time = 0;
  std::time_t verify_time = 0;
  int skip_len = (int)(match_list.size() / 200) + 1;
  // int skip_len = 1;
  int use_size = match_list.size() / skip_len;

  std::vector<size_t> index(use_size);
  std::vector<int> vote_list(use_size);
  for (size_t i = 0; i < index.size(); i++) {
    index[i] = i;
  }
  std::mutex mylock;
  auto t0 = std::chrono::high_resolution_clock::now();
  std::for_each(std::execution::par_unseq, index.begin(), index.end(),
                [&](const size_t &i) {
                  auto single_pair = match_list[i * skip_len];
                  int vote = 0;
                  Eigen::Matrix3d test_rot;
                  Eigen::Vector3d test_t;
                  triangle_solver(single_pair, test_rot, test_t);
                  for (size_t j = 0; j < match_list.size(); j++) {
                    auto verify_pair = match_list[j];
                    Eigen::Vector3d A = verify_pair.first.A;
                    Eigen::Vector3d A_transform = test_rot * A + test_t;
                    Eigen::Vector3d B = verify_pair.first.B;
                    Eigen::Vector3d B_transform = test_rot * B + test_t;
                    Eigen::Vector3d C = verify_pair.first.C;
                    Eigen::Vector3d C_transform = test_rot * C + test_t;
                    double dis_A = v3d_dis(A_transform, verify_pair.second.A);
                    double dis_B = v3d_dis(B_transform, verify_pair.second.B);
                    double dis_C = v3d_dis(C_transform, verify_pair.second.C);
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
  // std::cout << "max vote index:" << max_vote_index << " ,max vote:" <<
  // max_vote
  //           << std::endl;
  if (max_vote >= 4) {
    is_sucess = true;
    auto best_pair = match_list[max_vote_index * skip_len];
    int vote = 0;
    Eigen::Matrix3d test_rot;
    Eigen::Vector3d test_t;
    triangle_solver(best_pair, test_rot, test_t);
    rot = test_rot;
    t = test_t;
    for (size_t j = 0; j < match_list.size(); j++) {
      auto verify_pair = match_list[j];
      Eigen::Vector3d A = verify_pair.first.A;
      Eigen::Vector3d A_transform = test_rot * A + test_t;
      Eigen::Vector3d B = verify_pair.first.B;
      Eigen::Vector3d B_transform = test_rot * B + test_t;
      Eigen::Vector3d C = verify_pair.first.C;
      Eigen::Vector3d C_transform = test_rot * C + test_t;
      double dis_A = v3d_dis(A_transform, verify_pair.second.A);
      double dis_B = v3d_dis(B_transform, verify_pair.second.B);
      double dis_C = v3d_dis(C_transform, verify_pair.second.C);
      if (dis_A < dis_threshold && dis_B < dis_threshold &&
          dis_C < dis_threshold) {
        sucess_match_list.push_back(verify_pair);
      }
    }
  } else {
    is_sucess = false;
  }
}

void fine_loop_detection(
    std::vector<std::pair<Descriptor, Descriptor>> &match_list, bool &is_sucess,
    Eigen::Matrix3d &rot, Eigen::Vector3d &t,
    std::vector<std::pair<Descriptor, Descriptor>> &sucess_match_list,
    std::vector<std::pair<Descriptor, Descriptor>> &un_sucess_match_list) {
  sucess_match_list.clear();
  un_sucess_match_list.clear();
  // old 0.25,0.5 and 0.1
  // velodyne 0.5
  int skip_len = (int)(match_list.size() / 200) + 1;
  // old 2
  double dis_threshold = 5.0;
  is_sucess = false;
  int max_vote = 0;
  std::vector<std::pair<Descriptor, Descriptor>> unsucess_match_list;
  std::time_t solve_time = 0;
  std::time_t verify_time = 0;
  std::vector<Eigen::Vector3d> translation_list;
  std::vector<Eigen::Matrix3d> rotation_list;
  for (size_t i = 0; i < match_list.size(); i = i + skip_len) {
    auto single_pair = match_list[i];
    int vote = 0;
    Eigen::Matrix3d test_rot;
    Eigen::Vector3d test_t;
    std::vector<std::pair<Descriptor, Descriptor>> temp_match_list;
    std::vector<std::pair<Descriptor, Descriptor>> temp_unmatch_list;
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
    rot = rotation_list[best_index];
    t = translation_list[best_index];
    for (auto verify_pair : match_list) {
      Eigen::Vector3d A = verify_pair.first.A;
      Eigen::Vector3d A_transform = rot * A + t;
      Eigen::Vector3d B = verify_pair.first.B;
      Eigen::Vector3d B_transform = rot * B + t;
      Eigen::Vector3d C = verify_pair.first.C;
      Eigen::Vector3d C_transform = rot * C + t;
      double dis_A = v3d_dis(A_transform, verify_pair.second.A);
      double dis_B = v3d_dis(B_transform, verify_pair.second.B);
      double dis_C = v3d_dis(C_transform, verify_pair.second.C);
      if (dis_A < dis_threshold && dis_B < dis_threshold &&
          dis_C < dis_threshold) {
        sucess_match_list.push_back(verify_pair);
      } else {
        un_sucess_match_list.push_back(verify_pair);
      }
    }
  }
  if (sucess_match_list.size() >= 5) {
    is_sucess = true;
  } else {
    is_sucess = false;
  }
}

void getPlane(std::unordered_map<VOXEL_LOC, OctoTree *> &feat_map,
              pcl::PointCloud<pcl::PointXYZINormal>::Ptr &plane_cloud) {
  for (auto iter = feat_map.begin(); iter != feat_map.end(); iter++) {
    if (iter->second->plane_ptr_->is_plane) {
      pcl::PointXYZINormal pi;
      pi.x = iter->second->plane_ptr_->center[0];
      pi.y = iter->second->plane_ptr_->center[1];
      pi.z = iter->second->plane_ptr_->center[2];
      pi.normal_x = iter->second->plane_ptr_->normal[0];
      pi.normal_y = iter->second->plane_ptr_->normal[1];
      pi.normal_z = iter->second->plane_ptr_->normal[2];
      plane_cloud->push_back(pi);
    }
  }
}

void getPlaneCloud(std::unordered_map<VOXEL_LOC, OctoTree *> &feat_map,
                   pcl::PointCloud<pcl::PointXYZRGB>::Ptr &plane_cloud) {
  for (auto iter = feat_map.begin(); iter != feat_map.end(); iter++) {
    if (iter->second->plane_ptr_->is_plane) {
      for (size_t i = 0; i < iter->second->temp_cloud_->size(); i++) {
        pcl::PointXYZRGB pi;
        pi.x = iter->second->temp_cloud_->points[i].x;
        pi.y = iter->second->temp_cloud_->points[i].y;
        pi.z = iter->second->temp_cloud_->points[i].z;
        pi.r = fabs(iter->second->plane_ptr_->normal[0]) * 255;
        pi.g = fabs(iter->second->plane_ptr_->normal[1]) * 255;
        pi.b = fabs(iter->second->plane_ptr_->normal[2]) * 255;
        plane_cloud->push_back(pi);
      }
    }
  }
}

void getNoPlaneCloud(std::unordered_map<VOXEL_LOC, OctoTree *> &feat_map,
                     pcl::PointCloud<pcl::PointXYZI>::Ptr &no_plane_cloud) {
  for (auto iter = feat_map.begin(); iter != feat_map.end(); iter++) {
    if (!iter->second->plane_ptr_->is_plane) {
      for (size_t i = 0; i < iter->second->temp_cloud_->size(); i++) {
        pcl::PointXYZI pi;
        pi.x = iter->second->temp_cloud_->points[i].x;
        pi.y = iter->second->temp_cloud_->points[i].y;
        pi.z = iter->second->temp_cloud_->points[i].z;
        no_plane_cloud->push_back(pi);
      }
    }
  }
}

double
geometric_verify(const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &cloud,
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
  double normal_threshold = 0.2;
  double dis_threshold = 0.5;
  for (size_t i = 0; i < cloud->size(); i++) {
    pcl::PointXYZINormal searchPoint = cloud->points[i];
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
  // std::cout << "useful match num:" << useful_match << std::endl;
  // std::cout << "[Plane] match degree:" << useful_match / cloud->size()
  //           << std::endl;
  return useful_match / cloud->size();
}

void geometric_icp(
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
  double normal_threshold = 0.2;
  double dis_threshold = 0.5;
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
}

double icp_verify(const pcl::PointCloud<pcl::PointXYZ>::Ptr &key_cloud,
                  const pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr &kd_tree,
                  const Eigen::Matrix3d &rot, const Eigen::Vector3d t,
                  const double current_size) {
  int K = 1;
  int point_inc_num = 2;
  // old 0.5*0.5
  double dis_threshold = 1.0 * 1.0;
  // 创建两个向量，分别存放近邻的索引值、近邻的中心距
  std::vector<int> pointIdxNKNSearch(1);
  std::vector<float> pointNKNSquaredDistance(1);
  double match_num = 0;
  for (size_t i = 0; i < key_cloud->size(); i = i + point_inc_num) {
    pcl::PointXYZ searchPoint = key_cloud->points[i];
    Eigen::Vector3d pi(searchPoint.x, searchPoint.y, searchPoint.z);
    pi = rot * pi + t;
    searchPoint.x = pi[0];
    searchPoint.y = pi[1];
    searchPoint.z = pi[2];
    if (kd_tree->nearestKSearch(searchPoint, K, pointIdxNKNSearch,
                                pointNKNSquaredDistance) > 0) {
      if (pointNKNSquaredDistance[0] < dis_threshold) {
        match_num++;
      }
    }
  }
  // double match_degree = point_inc_num * match_num / history_cloud->size();
  double match_degree = point_inc_num * match_num / key_cloud->size();
  // std::cout << "match rate:" << match_degree << std::endl;
  return match_degree;
}

double ICP_verify(const pcl::PointCloud<pcl::PointXYZ>::Ptr &key_cloud,
                  const pcl::PointCloud<pcl::PointXYZ>::Ptr &history_cloud,
                  const Eigen::Matrix3d &rot, const Eigen::Vector3d t) {
  int point_inc_num = 1;
  pcl::KdTreeFLANN<pcl::PointXYZ> kd_tree;
  kd_tree.setInputCloud(history_cloud);
  int K = 1;
  // old 0.5*0.5
  double dis_threshold = 0.5 * 0.5;
  // 创建两个向量，分别存放近邻的索引值、近邻的中心距
  std::vector<int> pointIdxNKNSearch(1);
  std::vector<float> pointNKNSquaredDistance(1);
  double match_num = 0;
  for (size_t i = 0; i < key_cloud->size(); i = i + point_inc_num) {
    pcl::PointXYZ searchPoint = key_cloud->points[i];
    Eigen::Vector3d pi(searchPoint.x, searchPoint.y, searchPoint.z);
    pi = rot * pi + t;
    searchPoint.x = pi[0];
    searchPoint.y = pi[1];
    searchPoint.z = pi[2];
    if (kd_tree.nearestKSearch(searchPoint, K, pointIdxNKNSearch,
                               pointNKNSquaredDistance) > 0) {
      if (pointNKNSquaredDistance[0] < dis_threshold) {
        match_num++;
      }
    }
  }
  double match_degree = point_inc_num * match_num / key_cloud->size();
  std::cout << "[icp] match rate:" << match_degree << std::endl;
  return match_degree;
}

double ICP_verify(const pcl::PointCloud<pcl::PointXYZI>::Ptr &key_cloud,
                  const pcl::PointCloud<pcl::PointXYZI>::Ptr &history_cloud,
                  const Eigen::Matrix3d &rot, const Eigen::Vector3d t) {
  int point_inc_num = 1;
  pcl::KdTreeFLANN<pcl::PointXYZI> kd_tree;
  kd_tree.setInputCloud(history_cloud);
  int K = 1;
  // old 0.5*0.5
  double dis_threshold = 1 * 1;
  // 创建两个向量，分别存放近邻的索引值、近邻的中心距
  std::vector<int> pointIdxNKNSearch(1);
  std::vector<float> pointNKNSquaredDistance(1);
  double match_num = 0;
  for (size_t i = 0; i < key_cloud->size(); i = i + point_inc_num) {
    pcl::PointXYZI searchPoint = key_cloud->points[i];
    Eigen::Vector3d pi(searchPoint.x, searchPoint.y, searchPoint.z);
    pi = rot * pi + t;
    searchPoint.x = pi[0];
    searchPoint.y = pi[1];
    searchPoint.z = pi[2];
    if (kd_tree.nearestKSearch(searchPoint, K, pointIdxNKNSearch,
                               pointNKNSquaredDistance) > 0) {
      if (pointNKNSquaredDistance[0] < dis_threshold) {
        match_num++;
      }
    }
  }
  double match_degree = point_inc_num * match_num / key_cloud->size();
  std::cout << "[ICP] match rate:" << match_degree << std::endl;
  return match_degree;
}

double calcRMSE(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud1,
                const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud2,
                double max_dis) {
  pcl::KdTreeFLANN<pcl::PointXYZI> kd_tree;
  kd_tree.setInputCloud(cloud2);
  std::vector<int> pointIdxNKNSearch(1);
  std::vector<float> pointNKNSquaredDistance(1);
  double useful_num = 0;
  double rmse = 0;
  for (size_t i = 0; i < cloud1->points.size(); i++) {
    pcl::PointXYZI searchPoint = cloud1->points[i];
    if (kd_tree.nearestKSearch(searchPoint, 1, pointIdxNKNSearch,
                               pointNKNSquaredDistance) > 0) {
      if (pointNKNSquaredDistance[0] < max_dis * max_dis) {
        pcl::PointXYZI nearestPoinr = cloud2->points[pointIdxNKNSearch[0]];
        useful_num++;
        rmse += pow(nearestPoinr.x - searchPoint.x, 2) +
                pow(nearestPoinr.y - searchPoint.y, 2) +
                pow(nearestPoinr.z - searchPoint.z, 2);
      }
    }
  }
  rmse = sqrt(rmse / useful_num);
  // std::cout << "useful num:" << useful_num << " ,rmse:" << rmse << std::endl;
  return rmse;
}

Eigen::Vector4d PlaneFitting(const std::vector<Eigen::Vector3d> &plane_pts) {
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  for (const auto &pt : plane_pts)
    center += pt;
  center /= plane_pts.size();

  Eigen::MatrixXd A(plane_pts.size(), 3);
  for (int i = 0; i < plane_pts.size(); i++) {
    A(i, 0) = plane_pts[i][0] - center[0];
    A(i, 1) = plane_pts[i][1] - center[1];
    A(i, 2) = plane_pts[i][2] - center[2];
  }

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinV);
  const float a = svd.matrixV()(0, 2);
  const float b = svd.matrixV()(1, 2);
  const float c = svd.matrixV()(2, 2);
  const float d = -(a * center[0] + b * center[1] + c * center[2]);
  return Eigen::Vector4d(a, b, c, d);
}

double calcPlaneRMSE(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud1,
                     const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud2,
                     double max_dis) {
  pcl::KdTreeFLANN<pcl::PointXYZI> kd_tree;
  kd_tree.setInputCloud(cloud2);
  int near_num = 20;
  std::vector<int> pointIdxNKNSearch(20);
  std::vector<float> pointNKNSquaredDistance(20);
  double useful_num = 0;
  double rmse = 0;
  for (size_t i = 0; i < cloud1->points.size(); i++) {
    pcl::PointXYZI searchPoint = cloud1->points[i];
    if (kd_tree.nearestKSearch(searchPoint, 1, pointIdxNKNSearch,
                               pointNKNSquaredDistance) > 0) {
      std::vector<Eigen::Vector3d> point_list;
      for (size_t j = 0; j < near_num; j++) {
        Eigen::Vector3d pi(cloud2->points[pointIdxNKNSearch[j]].x,
                           cloud2->points[pointIdxNKNSearch[j]].y,
                           cloud2->points[pointIdxNKNSearch[j]].z);
        point_list.push_back(pi);
      }
      Eigen::Vector4d plane_params = PlaneFitting(point_list);
      double point_to_plane_dis =
          fabs(plane_params[0] * searchPoint.x +
               plane_params[1] * searchPoint.y +
               plane_params[2] * searchPoint.z + plane_params[3]) /
          (sqrt(pow(plane_params[0], 2) + pow(plane_params[1], 2) +
                pow(plane_params[2], 2)));
      if (point_to_plane_dis < max_dis * max_dis) {
        //   pcl::PointXYZI nearestPoinr =
        //   cloud2->points[pointIdxNKNSearch[0]];
        useful_num++;
        rmse += point_to_plane_dis;
        // }
      }
      // if (pointNKNSquaredDistance[0] < max_dis * max_dis) {
      //   pcl::PointXYZI nearestPoinr = cloud2->points[pointIdxNKNSearch[0]];
      //   useful_num++;
      //   rmse += pow(nearestPoinr.x - searchPoint.x, 2) +
      //           pow(nearestPoinr.y - searchPoint.y, 2) +
      //           pow(nearestPoinr.z - searchPoint.z, 2);
      // }
    }
  }
  rmse = sqrt(rmse / useful_num);
  std::cout << "useful num:" << useful_num << " ,rmse:" << rmse << std::endl;
  return rmse;
}

void max_constraint(const double &max_constraint_dis,
                    std::vector<BinaryDescriptor> &binary_list) {
  // max constraint
  pcl::PointCloud<pcl::PointXYZ>::Ptr prepare_key_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  pcl::KdTreeFLANN<pcl::PointXYZ> kd_tree;
  std::vector<int> pre_count_list;
  std::vector<bool> is_add_list;
  for (auto var : binary_list) {
    pcl::PointXYZ pi;
    pi.x = var.location[0];
    pi.y = var.location[1];
    pi.z = var.location[2];
    prepare_key_cloud->push_back(pi);
    pre_count_list.push_back(var.summary);
    is_add_list.push_back(true);
  }
  kd_tree.setInputCloud(prepare_key_cloud);
  std::vector<int> pointIdxRadiusSearch;
  std::vector<float> pointRadiusSquaredDistance;
  float radius = max_constraint_dis;
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

#endif