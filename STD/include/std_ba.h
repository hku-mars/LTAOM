#pragma once

#include <include/std.h>

typedef struct PlanePair {
  pcl::PointXYZINormal source_plane;
  pcl::PointXYZINormal target_plane;
  int source_id;
  int target_id;
} PlanePair;

//class PoseOptimizer {
//public:
//  PoseOptimizer(ConfigSetting config_setting)
//      : config_setting_(config_setting) {
//    loss_function_ = new ceres::HuberLoss(0.1);
//    // loss_function_ = nullptr;
//  }
//  void addConnection(double *pose1, double *pose2,
//                     std::pair<Eigen::Vector3d, Eigen::Matrix3d> &initial_guess,
//                     pcl::PointCloud<pcl::PointXYZINormal>::Ptr &plane_cloud1,
//                     pcl::PointCloud<pcl::PointXYZINormal>::Ptr &plane_cloud2,
//                     std::vector<PlanePair> &plane_pair_list);
//  void Solve();
//  ConfigSetting config_setting_;
//  ceres::Problem problem_;
//  ceres::Solver::Options options_;
//  ceres::Solver::Summary summary_;
//  ceres::LossFunction *loss_function_ = nullptr;

//  struct PlaneBaSolver {
//    PlaneBaSolver(const Eigen::Vector3d plane_normal1_,
//                  const Eigen::Vector3d plane_centriod1_,
//                  const Eigen::Vector3d plane_normal2_,
//                  const Eigen::Vector3d plane_centriod2_)
//        : plane_normal1(plane_normal1_), plane_centriod1(plane_centriod1_),
//          plane_normal2(plane_normal2_), plane_centriod2(plane_centriod2_){};

//    template <typename T>
//    bool operator()(const T *t1, const T *q1, const T *t2, const T *q2,
//                    T *residual) const {
//      const Eigen::Matrix<T, 3, 1> translation1{t1[0], t1[1], t1[2]};
//      const Eigen::Quaternion<T> quaternion1{q1[3], q1[0], q1[1], q1[2]};
//      const Eigen::Matrix<T, 3, 1> translation2{t2[0], t2[1], t2[2]};
//      const Eigen::Quaternion<T> quaternion2{q2[3], q2[0], q2[1], q2[2]};
//      Eigen::Matrix<T, 3, 1> plane_normal1_w{
//          T(plane_normal1[0]), T(plane_normal1[1]), T(plane_normal1[2])};
//      Eigen::Matrix<T, 3, 1> plane_centriod1_w{
//          T(plane_centriod1[0]), T(plane_centriod1[1]), T(plane_centriod1[2])};
//      Eigen::Matrix<T, 3, 1> plane_normal2_w{
//          T(plane_normal2[0]), T(plane_normal2[1]), T(plane_normal2[2])};
//      Eigen::Matrix<T, 3, 1> plane_centriod2_w{
//          T(plane_centriod2[0]), T(plane_centriod2[1]), T(plane_centriod2[2])};
//      plane_normal1_w = quaternion1 * plane_normal1_w;
//      plane_centriod1_w = quaternion1 * plane_centriod1_w + translation1;
//      plane_normal2_w = quaternion2 * plane_normal2_w;
//      plane_centriod2_w = quaternion2 * plane_centriod2_w + translation2;
//      residual[0] = plane_normal2_w.dot(plane_centriod1_w - plane_centriod2_w);
//      residual[1] = (plane_normal1_w - plane_normal2_w).norm();
//      return true;
//      //
//    }
//    static ceres::CostFunction *Create(const Eigen::Vector3d plane_normal1_,
//                                       const Eigen::Vector3d plane_centriod1_,
//                                       const Eigen::Vector3d plane_normal2_,
//                                       const Eigen::Vector3d plane_centriod2_) {
//      return (new ceres::AutoDiffCostFunction<PlaneBaSolver, 2, 3, 4, 3, 4>(
//          new PlaneBaSolver(plane_normal1_, plane_centriod1_, plane_normal2_,
//                            plane_centriod2_)));
//    }

//    Eigen::Vector3d plane_normal1;
//    Eigen::Vector3d plane_centriod1;
//    Eigen::Vector3d plane_normal2;
//    Eigen::Vector3d plane_centriod2;
//  };
//};

void geometric_icp(
    const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &cloud,
    const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &target_cloud,
    Eigen::Matrix3d &rot, Eigen::Vector3d &t);
