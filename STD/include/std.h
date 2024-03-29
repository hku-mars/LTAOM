#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <cv_bridge/cv_bridge.h>
#include <execution>
#include <fstream>
#include <mutex>
//#include <opencv/cv.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <pcl/common/io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ros/ros.h>
#include <sstream>
#include <stdio.h>
#include <string>
#include <unordered_map>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <vector>

#define HASH_P 116101
#define MAX_N 10000000000

typedef struct ConfigSetting {
  /* for point cloud pre-preocess*/
  int stop_skip_enable_ = 0;
  float ds_size_ = 0.5;
  int useful_corner_num_ = 30;
  int point_skip_ = 1;

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
  int line_filter_enable_ = 0;
  int touch_filter_enable_ = 0;

  /* for STD */
  float descriptor_near_num_ = 10;
  float descriptor_min_len_ = 1;
  float descriptor_max_len_ = 10;
  float non_max_suppression_radius_ = 3.0;
  float std_side_resolution_ = 0.2;

  /* for place recognition*/
  int skip_near_num_ = 20;
  int candidate_num_ = 50;
  int sub_frame_num_ = 10;
  float rough_dis_threshold_ = 0.03;
  float similarity_threshold_ = 0.7;
  float icp_threshold_ = 0.5;
  int ransac_Rt_thr = 4;
  float normal_threshold_ = 0.1;
  float dis_threshold_ = 0.3;

  /* for data base*/
  int std_add_skip_frame_ = 1;

  /* for result record*/
  int is_kitti_ = 1;
  /* extrinsic for lidar to vehicle*/
  Eigen::Matrix3d rot_lidar_to_vehicle_;
  Eigen::Vector3d t_lidar_to_vehicle_;

  /* for gt file style*/
  int gt_file_style_ = 0;

} ConfigSetting;

typedef struct BinaryDescriptor {
  std::vector<bool> occupy_array_;
  unsigned char summary_;
  Eigen::Vector3d location_;
} BinaryDescriptor;

typedef struct BinaryDescriptorF {
  // std::vector<bool> occupy_array_;
  // bool occupy_array_[49];
  unsigned char summary_;
  Eigen::Vector3f location_;
} BinaryDescriptorF;

// 1kb,12.8
typedef struct STD {
  Eigen::Vector3d triangle_;
  Eigen::Vector3d angle_;
  Eigen::Vector3d center_;
  unsigned short frame_number_;
  // std::vector<unsigned short> score_frame_;
  // std::vector<Eigen::Matrix3d> position_list_;
  BinaryDescriptor binary_A_;
  BinaryDescriptor binary_B_;
  BinaryDescriptor binary_C_;
} STD;

typedef struct STDF {
  Eigen::Vector3f triangle_;
  Eigen::Vector3f angle_;
  // Eigen::Vector3f A_;
  // Eigen::Vector3f B_;
  // Eigen::Vector3f C_;
  // unsigned char count_A_;
  // unsigned char count_B_;
  // unsigned char count_C_;
  Eigen::Vector3f center_;
  // float triangle_scale_;
  unsigned short frame_number_;
  // BinaryDescriptorF binary_A_;
  // BinaryDescriptorF binary_B_;
  // BinaryDescriptorF binary_C_;
} STDF;

typedef struct Plane {
  pcl::PointXYZINormal p_center_;
  Eigen::Vector3d center_;
  Eigen::Vector3d normal_;
  Eigen::Matrix3d covariance_;
  float radius_ = 0;
  float min_eigen_value_ = 1;
  float d_ = 0;
  int id_ = 0;
  int sub_plane_num_ = 0;
  int points_size_ = 0;
  bool is_plane_ = false;
} Plane;

typedef struct STDMatchList {
  std::vector<std::pair<STD, STD>> match_list_;
  int match_frame_;
  double mean_dis_;
} STDMatchList;

struct M_POINT {
  float xyz[3];
  float intensity;
  int count = 0;
};

class VOXEL_LOC {
public:
  int64_t x, y, z;

  VOXEL_LOC(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0)
      : x(vx), y(vy), z(vz) {}

  bool operator==(const VOXEL_LOC &other) const {
    return (x == other.x && y == other.y && z == other.z);
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

class STD_LOC {
public:
  int64_t x, y, z, a, b, c;

  STD_LOC(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0, int64_t va = 0,
          int64_t vb = 0, int64_t vc = 0)
      : x(vx), y(vy), z(vz), a(va), b(vb), c(vc) {}

  bool operator==(const STD_LOC &other) const {
    return (x == other.x && y == other.y && z == other.z);
    // return (x == other.x && y == other.y && z == other.z && a == other.a &&
    //         b == other.b && c == other.c);
  }
};

namespace std {
template <> struct hash<STD_LOC> {
  int64 operator()(const STD_LOC &s) const {
    using std::hash;
    using std::size_t;
    return ((((s.z) * HASH_P) % MAX_N + (s.y)) * HASH_P) % MAX_N + (s.x);
    // return ((((((((((s.z) * HASH_P) % MAX_N + (s.y)) * HASH_P) % MAX_N +
    //              (s.x)) *
    //             HASH_P) %
    //                MAX_N +
    //            s.a) *
    //           HASH_P) %
    //              MAX_N +
    //          s.b) *
    //         HASH_P) %
    //            MAX_N +
    //        s.c;
  }
};
} // namespace std

class OctoTree {
public:
  ConfigSetting config_setting_;
  std::vector<Eigen::Vector3d> voxel_points_;
  Plane *plane_ptr_;
  int layer_;
  int octo_state_; // 0 is end of tree, 1 is not
  int merge_num_ = 0;
  bool is_project_ = false;
  std::vector<Eigen::Vector3d> project_normal;
  bool is_publish_ = false;
  OctoTree *leaves_[8];
  double voxel_center_[3]; // x, y, z
  float quater_length_;
  bool init_octo_;
  OctoTree(const ConfigSetting &config_setting)
      : config_setting_(config_setting) {
    voxel_points_.clear();
    octo_state_ = 0;
    layer_ = 0;
    init_octo_ = false;
    for (int i = 0; i < 8; i++) {
      leaves_[i] = nullptr;
    }
    plane_ptr_ = new Plane;
  }
  void init_plane();
  void init_octo_tree();
};

void down_sampling_voxel(pcl::PointCloud<pcl::PointXYZI> &pl_feat,
                         double voxel_size);

void load_config_setting(std::string &config_file,
                         ConfigSetting &config_setting);

void load_pose_with_time(
    const std::string &pose_file,
    std::vector<std::pair<Eigen::Vector3d, Eigen::Matrix3d>> &pose_list,
    std::vector<double> &time_list);

void load_cu_pose_with_time(
    const std::string &pose_file,
    std::vector<std::pair<Eigen::Vector3d, Eigen::Matrix3d>> &pose_list,
    std::vector<double> &time_list);

void load_pose_with_frame(
    const std::string &pose_file,
    std::vector<std::pair<Eigen::Vector3d, Eigen::Matrix3d>> &pose_list,
    std::vector<int> &frame_number_list);

double binary_similarity(const BinaryDescriptor &b1,
                         const BinaryDescriptor &b2);

bool binary_greater_sort(BinaryDescriptor a, BinaryDescriptor b);
bool plane_greater_sort(Plane *plane1, Plane *plane2);

void init_voxel_map(const ConfigSetting &config_setting,
                    const pcl::PointCloud<pcl::PointXYZI> &input_cloud,
                    std::unordered_map<VOXEL_LOC, OctoTree *> &voxel_map);

void get_plane(std::unordered_map<VOXEL_LOC, OctoTree *> &feat_map,
               pcl::PointCloud<pcl::PointXYZINormal>::Ptr &plane_cloud);

void get_project_plane(const ConfigSetting &config_setting,
                       std::unordered_map<VOXEL_LOC, OctoTree *> &feat_map,

                       std::vector<Plane *> &project_plane_list);

void merge_plane(const ConfigSetting &config_setting,
                 std::vector<Plane *> &origin_list,
                 std::vector<Plane *> &merge_plane_list);

void non_max_suppression(const ConfigSetting &config_setting,
                         std::vector<BinaryDescriptor> &binary_list);

void binary_extractor(const ConfigSetting &config_setting,
                      const std::vector<Plane *> proj_plane_list,
                      const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
                      std::vector<BinaryDescriptor> &binary_descriptor_list);

void binary_extractor_debug(
    const ConfigSetting &config_setting,
    const std::vector<Plane *> proj_plane_list,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
    std::vector<BinaryDescriptor> &binary_descriptor_list,
    std::vector<BinaryDescriptor> &binary_descriptor_around_list);

void extract_binary(const ConfigSetting &config_setting,
                    const Eigen::Vector3d &project_center,
                    const Eigen::Vector3d &project_normal,
                    const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
                    std::vector<BinaryDescriptor> &binary_list);

void extract_binary_debug(
    const ConfigSetting &config_setting, const Eigen::Vector3d &project_center,
    const Eigen::Vector3d &project_normal,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
    std::vector<BinaryDescriptor> &binary_list,
    std::vector<BinaryDescriptor> &binary_around_list);

void extract_binary_all(const ConfigSetting &config_setting,
                        const Eigen::Vector3d &project_center,
                        const Eigen::Vector3d &project_normal,
                        const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
                        std::vector<BinaryDescriptor> &binary_list,
                        cv::Mat &binary_image);

void generate_std(const ConfigSetting &config_setting,
                  const std::vector<BinaryDescriptor> &binary_list,
                  const int &frame_id, std::vector<STD> &std_list);

void candidate_searcher(
    const ConfigSetting &config_setting,
    std::unordered_map<STD_LOC, std::vector<STD>> &descriptor_map,
    std::vector<STD> &current_STD_list,
    std::vector<STDMatchList> &alternative_match);

void candidate_searcher_old(
    const ConfigSetting &config_setting,
    std::unordered_map<STD_LOC, std::vector<STD>> &descriptor_map,
    std::vector<STD> &current_STD_list,
    std::vector<STDMatchList> &alternative_match);

void triangle_solver(std::pair<STD, STD> &std_pair, Eigen::Matrix3d &std_rot,
                     Eigen::Vector3d &std_t);

void fine_loop_detection(const ConfigSetting &config_setting,
                         std::vector<std::pair<STD, STD>> &match_list,
                         bool &fine_sucess, Eigen::Matrix3d &std_rot,
                         Eigen::Vector3d &std_t,
                         std::vector<std::pair<STD, STD>> &sucess_match_list,
                         std::vector<std::pair<STD, STD>> &unsucess_match_list);

void fine_loop_detection_tbb(const ConfigSetting &config_setting,
    std::vector<std::pair<STD, STD>> &match_list, bool &fine_sucess,
    Eigen::Matrix3d &std_rot, Eigen::Vector3d &std_t,
    std::vector<std::pair<STD, STD>> &sucess_match_list,
    std::vector<std::pair<STD, STD>> &unsucess_match_list, std::fstream &debug_file);

double
geometric_verify(const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &source_cloud,
                 const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &target_cloud,
                 const Eigen::Matrix3d &rot, const Eigen::Vector3d &t);

double
geometric_verify(const ConfigSetting &config_setting,
                 const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &source_cloud,
                 const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &target_cloud,
                 const Eigen::Matrix3d &rot, const Eigen::Vector3d &t);

double geometric_icp(const ConfigSetting &config_setting,
                     const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &cloud,
                     const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &target_cloud,
                     Eigen::Matrix3d &rot, Eigen::Vector3d &t);

void add_STD(std::unordered_map<STD_LOC, std::vector<STD>> &descriptor_map,
             std::vector<STD> &STD_list);

void publish_std(const std::vector<std::pair<STD, STD>> &match_std_list,
                 const ros::Publisher &std_publisher);

void publish_std_list(const std::vector<STD> &std_list,
                      const ros::Publisher &std_publisher);

void publish_binary(const std::vector<BinaryDescriptor> &binary_list,
                    const Eigen::Vector3d &text_color,
                    const std::string &text_ns,
                    const ros::Publisher &text_publisher);

double
calc_triangle_dis(const std::vector<std::pair<STD, STD>> &match_std_list);
double
calc_binary_similaity(const std::vector<std::pair<STD, STD>> &match_std_list);

double calc_overlap(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud1,
                    const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud2,
                    double dis_threshold);

double calc_overlap(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud1,
                    const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud2,
                    double dis_threshold, int skip_num);

void CalcQuation(const Eigen::Vector3d &vec, const int axis,
                 geometry_msgs::Quaternion &q);

void pubPlane(const ros::Publisher &plane_pub, const std::string plane_ns,
              const int plane_id, const pcl::PointXYZINormal normal_p,
              const float radius, const Eigen::Vector3d rgb);

pcl::PointXYZI vec2point(const Eigen::Vector3d &vec);
Eigen::Vector3d point2vec(const pcl::PointXYZI &pi);

Eigen::Vector3d normal2vec(const pcl::PointXYZINormal &pi);

template <typename T> Eigen::Vector3d point2vec(const T &pi) {
  Eigen::Vector3d vec(pi.x, pi.y, pi.z);
  return vec;
}

double time_inc(std::chrono::_V2::system_clock::time_point &t_end,
                std::chrono::_V2::system_clock::time_point &t_begin);

void load_array(std::vector<bool> &occupy_array_, const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_to_load, const int i);
void save_an_array(const BinaryDescriptor& bdes,pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_to_store);

void save_descriptor(const std::vector<STD> &current_descriptor,
                     pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_to_store);
void load_descriptor(std::unordered_map<STD_LOC, std::vector<STD>>  &feat_map,
                     const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_to_load, int &index_max);
