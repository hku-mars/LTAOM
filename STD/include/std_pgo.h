#pragma once

#include <include/std.h>

struct Pose3d {
  Eigen::Vector3d p;
  Eigen::Quaterniond q;

  // The name of the data type in the g2o file format.
  static std::string name() { return "VERTEX_SE3:QUAT"; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

inline std::istream &operator>>(std::istream &input, Pose3d &pose) {
  input >> pose.p.x() >> pose.p.y() >> pose.p.z() >> pose.q.x() >> pose.q.y() >>
      pose.q.z() >> pose.q.w();
  // Normalize the quaternion to account for precision loss due to
  // serialization.
  pose.q.normalize();
  return input;
}

using MapOfPoses =
    std::map<int, Pose3d, std::less<int>,
             Eigen::aligned_allocator<std::pair<const int, Pose3d>>>;

// The constraint between two vertices in the pose graph. The constraint is the
// transformation from vertex id_begin to vertex id_end.
struct Constraint3d {
  int id_begin;
  int id_end;

  // The transformation that represents the pose of the end frame E w.r.t. the
  // begin frame B. In other words, it transforms a vector in the E frame to
  // the B frame.
  Pose3d t_be;

  // The inverse of the covariance matrix for the measurement. The order of the
  // entries are x, y, z, delta orientation.
  Eigen::Matrix<double, 6, 6> information;

  // The name of the data type in the g2o file format.
  static std::string name() { return "EDGE_SE3:QUAT"; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

inline std::istream &operator>>(std::istream &input, Constraint3d &constraint) {
  Pose3d &t_be = constraint.t_be;
  input >> constraint.id_begin >> constraint.id_end >> t_be;

  for (int i = 0; i < 6 && input.good(); ++i) {
    for (int j = i; j < 6 && input.good(); ++j) {
      input >> constraint.information(i, j);
      if (i != j) {
        constraint.information(j, i) = constraint.information(i, j);
      }
    }
  }
  return input;
}

using VectorOfConstraints =
    std::vector<Constraint3d, Eigen::aligned_allocator<Constraint3d>>;

class PoseGraph3dErrorTerm {
public:
  PoseGraph3dErrorTerm(Pose3d t_ab_measured,
                       Eigen::Matrix<double, 6, 6> sqrt_information)
      : t_ab_measured_(std::move(t_ab_measured)),
        sqrt_information_(std::move(sqrt_information)) {}

  template <typename T>
  bool operator()(const T *const p_a_ptr, const T *const q_a_ptr,
                  const T *const p_b_ptr, const T *const q_b_ptr,
                  T *residuals_ptr) const {
    // Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_a(p_a_ptr);
    // Eigen::Map<const Eigen::Quaternion<T>> q_a(q_a_ptr);

    // Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_b(p_b_ptr);
    // Eigen::Map<const Eigen::Quaternion<T>> q_b(q_b_ptr);

    const Eigen::Matrix<T, 3, 1> p_a{p_a_ptr[0], p_a_ptr[1], p_a_ptr[2]};
    const Eigen::Quaternion<T> q_a{q_a_ptr[3], q_a_ptr[0], q_a_ptr[1],
                                   q_a_ptr[2]};
    const Eigen::Matrix<T, 3, 1> p_b{p_b_ptr[0], p_b_ptr[1], p_b_ptr[2]};
    const Eigen::Quaternion<T> q_b{q_b_ptr[3], q_b_ptr[0], q_b_ptr[1],
                                   q_b_ptr[2]};

    // Compute the relative transformation between the two frames.
    Eigen::Quaternion<T> q_a_inverse = q_a.conjugate();
    Eigen::Quaternion<T> q_ab_estimated = q_a_inverse * q_b;

    // Represent the displacement between the two frames in the A frame.
    Eigen::Matrix<T, 3, 1> p_ab_estimated = q_a_inverse * (p_b - p_a);

    // Compute the error between the two orientation estimates.
    Eigen::Quaternion<T> delta_q =
        t_ab_measured_.q.template cast<T>() * q_ab_estimated.conjugate();

    // Compute the residuals.
    // [ position         ]   [ delta_p          ]
    // [ orientation (3x1)] = [ 2 * delta_q(0:2) ]
    Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals(residuals_ptr);
    residuals.template block<3, 1>(0, 0) =
        p_ab_estimated - t_ab_measured_.p.template cast<T>();
    residuals.template block<3, 1>(3, 0) = T(2.0) * delta_q.vec();
    // Scale the residuals by the measurement uncertainty.
    residuals.applyOnTheLeft(sqrt_information_.template cast<T>());

    return true;
  }

  static ceres::CostFunction *
  Create(const Pose3d &t_ab_measured,
         const Eigen::Matrix<double, 6, 6> &sqrt_information) {
    return new ceres::AutoDiffCostFunction<PoseGraph3dErrorTerm, 6, 3, 4, 3, 4>(
        new PoseGraph3dErrorTerm(t_ab_measured, sqrt_information));
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  // The measurement for the position of B relative to A in the A frame.
  const Pose3d t_ab_measured_;
  // The square root of the measurement information matrix.
  const Eigen::Matrix<double, 6, 6> sqrt_information_;
};

class PoseGraph3dErrorTermOwn {
public:
  PoseGraph3dErrorTermOwn(const Eigen::Vector3d &t_ab_,
                          const Eigen::Quaterniond &q_ab_,
                          Eigen::Matrix<double, 6, 6> sqrt_information)
      : t_ab(t_ab_), q_ab(q_ab_),
        sqrt_information_(std::move(sqrt_information)) {}

  template <typename T>
  bool operator()(const T *const p_a_ptr, const T *const q_a_ptr,
                  const T *const p_b_ptr, const T *const q_b_ptr,
                  T *residuals_ptr) const {

    const Eigen::Matrix<T, 3, 1> p_a{p_a_ptr[0], p_a_ptr[1], p_a_ptr[2]};
    const Eigen::Quaternion<T> q_a{q_a_ptr[3], q_a_ptr[0], q_a_ptr[1],
                                   q_a_ptr[2]};
    const Eigen::Matrix<T, 3, 1> p_b{p_b_ptr[0], p_b_ptr[1], p_b_ptr[2]};
    const Eigen::Quaternion<T> q_b{q_b_ptr[3], q_b_ptr[0], q_b_ptr[1],
                                   q_b_ptr[2]};

    // Compute the relative transformation between the two frames.
    Eigen::Quaternion<T> q_a_inverse = q_a.conjugate();
    Eigen::Quaternion<T> q_ab_estimated = q_a_inverse * q_b;

    // Represent the displacement between the two frames in the A frame.
    Eigen::Matrix<T, 3, 1> p_ab_estimated = q_a_inverse * (p_b - p_a);

    // Compute the error between the two orientation estimates.
    Eigen::Quaternion<T> delta_q =
        q_ab.template cast<T>() * q_ab_estimated.conjugate();

    // Compute the residuals.
    // [ position         ]   [ delta_p          ]
    // [ orientation (3x1)] = [ 2 * delta_q(0:2) ]
    Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals(residuals_ptr);
    residuals[0] = p_ab_estimated[0] - T(t_ab[0]);
    residuals[1] = p_ab_estimated[1] - T(t_ab[1]);
    residuals[2] = p_ab_estimated[2] - T(t_ab[2]);
    residuals.template block<3, 1>(0, 0) =
        p_ab_estimated - t_ab.template cast<T>();
    residuals.template block<3, 1>(3, 0) = T(2.0) * delta_q.vec();
    // std::cout << "p_ab_estimated:" << p_ab_estimated
    //           << ", t_ab:" << t_ab.transpose() << std::endl;
    // std::cout << "residual:" << residuals << std::endl;
    // getchar();
    // Scale the residuals by the measurement
    // uncertainty.
    residuals.applyOnTheLeft(sqrt_information_.template cast<T>());
    return true;
  }

  static ceres::CostFunction *
  Create(const Eigen::Vector3d &t_ab, const Eigen::Quaterniond &q_ab,
         const Eigen::Matrix<double, 6, 6> &sqrt_information) {
    return new ceres::AutoDiffCostFunction<PoseGraph3dErrorTermOwn, 3, 3, 4, 3,
                                           4>(
        new PoseGraph3dErrorTermOwn(t_ab, q_ab, sqrt_information));
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  // The measurement for the position of B relative to A in the A frame.
  const Eigen::Vector3d t_ab;
  const Eigen::Quaterniond q_ab;
  // The square root of the measurement information matrix.
  const Eigen::Matrix<double, 6, 6> sqrt_information_;
};

void geometric_icp(
    const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &cloud,
    const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &target_cloud,
    Eigen::Matrix3d &rot, Eigen::Vector3d &t);

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