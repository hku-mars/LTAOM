#ifndef PREDEFINED_TYPES_H
#define PREDEFINED_TYPES_H

#include <cmath>

#include <pcl/point_types.h>
#include <gtsam/nonlinear/ISAM2.h>

typedef pcl::PointXYZINormal PointType;

inline double rad2deg(double radians)
{
  return radians * 180.0 / M_PI;
}

inline double deg2rad(double degrees)
{
  return degrees * M_PI / 180.0;
}

struct Pose6D {
  double x;
  double y;
  double z;
  double roll;
  double pitch;
  double yaw;
};

class InterFrameFactors
{
public:
  InterFrameFactors() {}
  InterFrameFactors(const gtsam::FastVector<size_t> &factors_toremove_in, const std::vector<std::pair<int,int>> odomfactor_keys_in) {
    factors_toremove = factors_toremove_in;
    odomfactor_keys = odomfactor_keys_in;
  }

  ~InterFrameFactors() {}

  gtsam::FastVector<size_t> factors_toremove;
  std::vector<std::pair<int,int>> odomfactor_keys;
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

#define HASH_P 116101
#define MAX_N 10000000000
// Hash value
namespace std {
template <> struct hash<VOXEL_LOC> {
  int64_t operator()(const VOXEL_LOC &s) const {
    using std::hash;
    using std::size_t;
    return s.x*1e12 + s.y*1e6 + s.z;
  }
};

} // namespace std


#endif // PREDEFINED_TYPES_H
