#ifndef EXTRA_LIB_H
#define EXTRA_LIB_H
#include "common_lib.h"
#include "ikd-Tree/ikd_Tree.h"
#include "use-ikfom.hpp"

#include <ros/ros.h>
#include <std_msgs/Float32MultiArray.h>
#include <visualization_msgs/MarkerArray.h>
#include <pcl/registration/icp.h>

#include <unordered_map>

#include <opencv2/opencv.hpp>
#include <cmath>

struct  SubmapInfo
{
     SubmapInfo()
    {
        msg_time = 0.0;
        oriPoseSet = false;
        corPoseSet = false;
//        this->lidar.reset(new PointCloudXYZI());
    };
    double msg_time;
    int submap_index;
//    PointCloudXYZI::Ptr lidar;
    PointVector cloud_ontree;
//    PointVector boundry_ptx2; //[x_max,y_max,z_max],[x_min,y_min,z_min];
//    geometry_msgs::Pose lidar_pose;
//    geometry_msgs::Pose corrected_pose;
    M3D lidar_pose_rotM;
    V3D lidar_pose_tran;
    M3D corr_pose_rotM;
    V3D corr_pose_tran;
    M3D offset_R_L_I;
    V3D offset_t_L_I;
    bool oriPoseSet, corPoseSet;
    std::vector<std::tuple<double, Eigen::Matrix3d, Eigen::Vector3d>> scan_poses_insubmap;
    std::vector<std::tuple<double, Eigen::Matrix3d, Eigen::Vector3d>> extrinsics_insubmap;
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// Key of hash table
class VOXEL_LOC
{
public:
  int64_t x, y, z;

  VOXEL_LOC(int64_t vx=0, int64_t vy=0, int64_t vz=0): x(vx), y(vy), z(vz){}

  bool operator== (const VOXEL_LOC &other) const
  {
    return (x==other.x && y==other.y && z==other.z);
  }
};
// Hash value
namespace std
{
template<>
struct hash<VOXEL_LOC>
{
  size_t operator() (const VOXEL_LOC &s) const
  {
    using std::size_t; using std::hash;
    return ((hash<int64_t>()(s.x) ^ (hash<int64_t>()(s.y) << 1)) >> 1) ^ (hash<int64_t>()(s.z) << 1);
  }
};
}

class OCTO_TREE_NEW
{
public:
  pcl::PointCloud<PointType>::Ptr plvec_pcl;
  int octo_state; // 0 is end of tree, 1 is not
  int ftype;
  int points_size;
  bool is2opt;
  double feat_eigen_ratio_20, feat_eigen_ratio_21;
  PointType ap_centor_direct;
  double voxel_center[3]; // x, y, z
  double quater_length;
  OCTO_TREE_NEW* leaves[8];
  int layer_lowest;
  pcl::PointCloud<PointType> root_centors;

  OCTO_TREE_NEW()
  {
    octo_state = 0;
    for(int i=0; i<8; i++)
    {
      leaves[i] = nullptr;
    }

    is2opt = true;
    plvec_pcl.reset(new pcl::PointCloud<PointType>());
  }
};

namespace ExtraLib {

void pubCorrectionIds(const ros::Publisher &pub_handle, const V3D &pos, const int id);

bool GetOneLineAndSplitByComma(std::istream& fptr, std::vector<std::string> &out_str);

//bool refineWithPt2PtICP(const PointCloudXYZI::Ptr &feats_down_body, const float rmse_thr, const int iteration_thr, const boost::shared_ptr<KD_TREE> &ikd_in, \
//                           M3D &Ricp, V3D &ticp,  const M3D &Rguess, const V3D &tguess, const int pos_id_lc,  std::ostream &fout_dbg);

//bool refineWithPt2PlaneICP(const PointCloudXYZI::Ptr &feats_down_body, const float rmse_thr, const int iteration_thr, const boost::shared_ptr<KD_TREE> &ikd_in, \
//                           M3D &Ricp, V3D &ticp,  const M3D &Rguess, const V3D &tguess, const int pos_id_lc,  std::ostream &fout_dbg);

//void setKFPose(esekfom::esekf<state_ikfom, 12, input_ikfom> & kf, state_ikfom &tmp_state, const boost::shared_ptr<KD_TREE> &ikd_in, \
//               const PointCloudXYZI::Ptr &feats_down_body, const MD(4,4) &Tcomb, const MD(4,4) &Tcomb_nooff, const int pos_id_lc,\
//               std_msgs::Float32MultiArrayPtr &notification_msg, std_msgs::Float64MultiArrayPtr &notification_msg2, std::ostream &fout_dbg);

V3D esti_center(const PointVector &point_near);

void eigenRtToPoseMsg(const M3D &R, const V3D &t, geometry_msgs::Pose &out);

M3D eulToRotM(double roll, double pitch, double yaw);

void poseMsgToEigenRT(const geometry_msgs::Pose &m, M3D &R, V3D &t);

V3D geometryOrientationToRPY(const geometry_msgs::Quaternion pose_in);

void printInfo(std::unordered_map<int, SubmapInfo> &unmap_submap_info, ostream& fout_dbg);

PointVector findSubmapBoundingBoxPt(const PointVector &submap_cloud);

float calcScanSubmapOverlapRatio(const PointType scan_pt_max, const PointType scan_pt_min, const PointType submap_pt_max, const PointType submap_pt_min);

void CutVoxel3d(std::unordered_map<VOXEL_LOC, OCTO_TREE_NEW*> &feat_map,
                const pcl::PointCloud<PointType>::Ptr  pl_feat, float voxel_box_size);

cv::Mat CreateImage(std::unordered_map<VOXEL_LOC, OCTO_TREE_NEW*> &voxels_uomap_flat,\
                    std::unordered_map<VOXEL_LOC, OCTO_TREE_NEW*> &voxels_uomap_flat2,\
                    const float &v_size, const bool &to_print);
}


#endif // EXTRA_LIB_H
