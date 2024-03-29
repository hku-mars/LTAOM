#include "include/std_ba.h"
#include "include/std.h"
#include "include/std_pgo.h"
#include <nav_msgs/Odometry.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud2.h>

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
//  ceres::Manifold *quaternion_manifold = new ceres::EigenQuaternionManifold;
  ceres::LocalParameterization *q_parameterization =
      new ceres::EigenQuaternionParameterization();
  ceres::Problem problem;
  ceres::LossFunction *loss_function = nullptr; // new ceres::HuberLoss(0.1);
  double para_q[4] = {q.x(), q.y(), q.z(), q.w()};
  double para_t[3] = {t(0), t(1), t(2)};
//  problem.AddParameterBlock(para_q, 4, quaternion_manifold);
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
      double point_to_point_dis = (pi - tpi).norm();
      double point_to_plane = fabs(tni.transpose() * (pi - tpi));
      if ((normal_inc.norm() < normal_threshold ||
           normal_add.norm() < normal_threshold) &&
          point_to_plane < dis_threshold && point_to_point_dis < 3) {
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
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.max_num_iterations = 100;
  options.minimizer_progress_to_stdout = false;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  Eigen::Quaterniond q_opt(para_q[3], para_q[0], para_q[1], para_q[2]);
  // std::cout << summary.BriefReport() << std::endl;
  rot = q_opt.toRotationMatrix();
  t << t_last_curr(0), t_last_curr(1), t_last_curr(2);

  std::cout << "useful match for icp:" << useful_match << std::endl;
}

//void PoseOptimizer::addConnection(
//    double *pose1, double *pose2,
//    std::pair<Eigen::Vector3d, Eigen::Matrix3d> &initial_guess,
//    pcl::PointCloud<pcl::PointXYZINormal>::Ptr &plane_cloud1,
//    pcl::PointCloud<pcl::PointXYZINormal>::Ptr &plane_cloud2,
//    std::vector<PlanePair> &plane_match_list) {
//  Eigen::Vector3d translation1(pose1[0], pose1[1], pose1[2]);
//  Eigen::Quaterniond quaternion1(pose1[6], pose1[3], pose1[4], pose1[5]);
//  Eigen::Matrix3d rotation1 = quaternion1.toRotationMatrix();
//  Eigen::Vector3d translation2(pose2[0], pose2[1], pose2[2]);
//  Eigen::Quaterniond quaternion2(pose2[6], pose2[3], pose2[4], pose2[5]);
//  Eigen::Matrix3d rotation2 = quaternion2.toRotationMatrix();
//  ceres::Manifold *quaternion_manifold = new ceres::EigenQuaternionManifold;
//  problem_.AddParameterBlock(pose1, 3);
//  problem_.AddParameterBlock(pose1 + 3, 4, quaternion_manifold);
//  problem_.AddParameterBlock(pose2, 3);
//  problem_.AddParameterBlock(pose2 + 3, 4, quaternion_manifold);
//  int useful_num = 0;
//  pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kd_tree(
//      new pcl::KdTreeFLANN<pcl::PointXYZ>);
//  pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(
//      new pcl::PointCloud<pcl::PointXYZ>);
//  for (size_t i = 0; i < plane_cloud2->size(); i++) {
//    pcl::PointXYZ pi;
//    pi.x = plane_cloud2->points[i].x;
//    pi.y = plane_cloud2->points[i].y;
//    pi.z = plane_cloud2->points[i].z;
//    input_cloud->push_back(pi);
//  }
//  kd_tree->setInputCloud(input_cloud);
//  for (size_t i = 0; i < plane_cloud1->size(); i++) {
//    pcl::PointXYZINormal searchPoint = plane_cloud1->points[i];
//    Eigen::Vector3d pi(searchPoint.x, searchPoint.y, searchPoint.z);
//    Eigen::Vector3d ni(searchPoint.normal_x, searchPoint.normal_y,
//                       searchPoint.normal_z);
//    Eigen::Vector3d guess_pi = initial_guess.second * pi + initial_guess.first;
//    Eigen::Vector3d guess_ni = initial_guess.second * ni;
//    pcl::PointXYZ useSearchPoint;
//    useSearchPoint.x = guess_pi[0];
//    useSearchPoint.y = guess_pi[1];
//    useSearchPoint.z = guess_pi[2];

//    std::vector<int> pointIdxNKNSearch(1);
//    std::vector<float> pointNKNSquaredDistance(1);
//    if (kd_tree->nearestKSearch(useSearchPoint, 1, pointIdxNKNSearch,
//                                pointNKNSquaredDistance) > 0) {
//      pcl::PointXYZINormal nearstPoint =
//          plane_cloud2->points[pointIdxNKNSearch[0]];
//      Eigen::Vector3d tpi(nearstPoint.x, nearstPoint.y, nearstPoint.z);
//      Eigen::Vector3d tni(nearstPoint.normal_x, nearstPoint.normal_y,
//                          nearstPoint.normal_z);
//      Eigen::Vector3d normal_inc = guess_ni - tni;
//      Eigen::Vector3d normal_add = guess_ni + tni;
//      double point_to_point_dis = (guess_pi - tpi).norm();
//      double point_to_plane = fabs(tni.transpose() * (guess_pi - tpi));
//      if ((normal_inc.norm() < config_setting_.normal_threshold_ ||
//           normal_add.norm() < config_setting_.normal_threshold_) &&
//          point_to_plane < config_setting_.dis_threshold_ &&
//          point_to_point_dis < 3) {
//        useful_num++;
//        ceres::CostFunction *cost_function;
//        Eigen::Vector3d plane_normal1 = rotation1.transpose() * ni;
//        Eigen::Vector3d plane_centriod1 =
//            rotation1.transpose() * pi - rotation1.transpose() * translation1;
//        Eigen::Vector3d plane_normal2 = rotation2.transpose() * tni;
//        Eigen::Vector3d plane_centriod2 =
//            rotation2.transpose() * tpi - rotation2.transpose() * translation2;
//        cost_function = PlaneBaSolver::Create(plane_normal1, plane_centriod1,
//                                              plane_normal2, plane_centriod2);
//        problem_.AddResidualBlock(cost_function, loss_function_, pose1,
//                                  pose1 + 3, pose2, pose2 + 3);
//        PlanePair pp;
//        pp.source_plane.x = plane_centriod1[0];
//        pp.source_plane.y = plane_centriod1[1];
//        pp.source_plane.z = plane_centriod1[2];
//        pp.source_plane.normal_x = plane_normal1[0];
//        pp.source_plane.normal_y = plane_normal1[1];
//        pp.source_plane.normal_z = plane_normal1[2];

//        pp.target_plane.x = plane_centriod2[0];
//        pp.target_plane.y = plane_centriod2[1];
//        pp.target_plane.z = plane_centriod2[2];
//        pp.target_plane.normal_x = plane_normal2[0];
//        pp.target_plane.normal_y = plane_normal2[1];
//        pp.target_plane.normal_z = plane_normal2[2];
//        plane_match_list.push_back(pp);
//      }
//    }
//  }
//  std::cout << "useful num:" << useful_num << std::endl;
//}

//void PoseOptimizer::Solve() {
//  // sparse surb
//  options_.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
//  options_.max_num_iterations = 100;
//  options_.minimizer_progress_to_stdout = false;
//  ceres::Solve(options_, &problem_, &summary_);
//  std::cout << summary_.FullReport() << '\n';
//}
