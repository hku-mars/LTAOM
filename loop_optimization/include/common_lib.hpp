#ifndef COMMON_LIB_HPP
#define COMMON_LIB_HPP

#include "predefined_types.h"

#include <nav_msgs/Odometry.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/registration/icp.h>

#include <gtsam/geometry/Pose3.h>
#include <tf/tf.h>

bool GetOneLineAndSplitByComma(std::istream& fptr, std::vector<std::string> &out_str)
{
    out_str.clear();
    std::string cell;
    std::string line;
    std::getline(fptr,line);
    std::stringstream lineStream(line);
    while(std::getline(lineStream,cell, ','))
        out_str.push_back(cell);

    if (out_str.empty()) return false;
    return true;
}
Eigen::Quaterniond EulerToEigenQuat(double roll, double pitch, double yaw){
  double c1 = cos(roll*0.5);
  double s1 = sin(roll*0.5);
  double c2 = cos(pitch*0.5);
  double s2 = sin(pitch*0.5);
  double c3 = cos(yaw*0.5);
  double s3 = sin(yaw*0.5);
  return Eigen::Quaterniond(c1*c2*c3 - s1*s2*s3, s1*c2*c3 + c1*s2*s3, -s1*c2*s3 + c1*s2*c3, c1*c2*s3 + s1*s2*c3);
}
Eigen::Matrix3d EulerToRotM(double roll, double pitch, double yaw){
  double cx = cos(roll);
  double sx = sin(roll);
  double cy = cos(pitch);
  double sy = sin(pitch);
  double cz = cos(yaw);
  double sz = sin(yaw);
  Eigen::Matrix3d R;
  R <<           cy*cz,           -cy*sz,     sy,
       cx*sz + cz*sx*sy, cx*cz - sx*sy*sz, -cy*sx,
       sx*sz - cx*cz*sy, cz*sx + cx*sy*sz,  cx*cy;
  return R;
}
Eigen::Matrix3d QuatToRotM(double w, double x, double y, double z){
  Eigen::Matrix3d R;
  R <<  1 - 2*(y*y + z*z),   2*(x*y - w*z),   2*(x*z + w*y),
      2*(x*y + w*z), 1 - 2*(x*x + z*z),   2*(y*z - w*x),
      2*(x*z - w*y),   2*(y*z + w*x), 1 - 2*(x*x + y*y);
  return R;
}
Pose6D OdomMsgToPose6D(const nav_msgs::Odometry::ConstPtr &odom_msg)
{
  auto x = odom_msg->pose.pose.position.x;
  auto y = odom_msg->pose.pose.position.y;
  auto z = odom_msg->pose.pose.position.z;
  auto qx = odom_msg->pose.pose.orientation.x;
  auto qy = odom_msg->pose.pose.orientation.y;
  auto qz = odom_msg->pose.pose.orientation.z;
  auto qw = odom_msg->pose.pose.orientation.w;
  double roll, pitch, yaw;
  tf::Matrix3x3(tf::Quaternion(qx, qy, qz, qw)).getRPY(roll, pitch, yaw);
  return Pose6D{x, y, z, roll, pitch, yaw};
}
gtsam::Pose3 GeoPoseMsgToGTSPose(const geometry_msgs::Pose& pose)
{
  auto x = pose.position.x;
  auto y = pose.position.y;
  auto z = pose.position.z;
  auto qx = pose.orientation.x;
  auto qy = pose.orientation.y;
  auto qz = pose.orientation.z;
  auto qw = pose.orientation.w;
  double roll, pitch, yaw;
  tf::Matrix3x3(tf::Quaternion(qx, qy, qz, qw)).getRPY(roll, pitch, yaw);
  return gtsam::Pose3(gtsam::Rot3::RzRyRx(roll, pitch, yaw), gtsam::Point3(x, y, z));
}
gtsam::Pose3 Pose6DToGTSPose(const Pose6D& p)
{
  return gtsam::Pose3(gtsam::Rot3::RzRyRx(p.roll, p.pitch, p.yaw), gtsam::Point3(p.x, p.y, p.z));
}
void Pose6DToEigenRT(const Pose6D& p, gtsam::Matrix3 &R, gtsam::Vector3 &t)
{
  R = gtsam::Rot3::RzRyRx(p.roll, p.pitch, p.yaw).matrix();
  t = gtsam::Point3(p.x, p.y, p.z).vector();
}
Eigen::Matrix4f GeoPoseMsgToEigenM4f(const geometry_msgs::PoseWithCovarianceConstPtr& lc_msg){
  auto x = lc_msg->pose.position.x;
  auto y = lc_msg->pose.position.y;
  auto z = lc_msg->pose.position.z;
  auto qx = lc_msg->pose.orientation.x;
  auto qy = lc_msg->pose.orientation.y;
  auto qz = lc_msg->pose.orientation.z;
  auto qw = lc_msg->pose.orientation.w;
//  Eigen::Quaterniond q_2(qx, qy, qz, qw);
//  Eigen::Matrix3f R = q_2.toRotationMatrix().cast<float>();
  Eigen::Matrix3f R = QuatToRotM(qw, qx, qy, qz).cast<float>();
  Eigen::Vector3f t(x,y,z);
  Eigen::Matrix4f out = Eigen::Matrix4f::Identity();
  out.block<3,3>(0,0) = R;
  out.block<3,1>(0,3) = t;
  return out;
}
Eigen::Matrix4f GTSPoseToEigenM4f(const gtsam::Pose3& pose){
  gtsam::Point3 t = pose.translation();
  gtsam::Rot3 R = pose.rotation();
  auto col1 = R.column(1); // Point3
  auto col2 = R.column(2); // Point3
  auto col3 = R.column(3); // Point3

  Eigen::Matrix4d out = Eigen::Matrix4d::Identity();
  out << col1.x(), col2.x(), col3.x(), t.x(),
         col1.y(), col2.y(), col3.y(), t.y(),
         col1.z(), col2.z(), col3.z(), t.z(),
         0,        0,        0,        1;
  return out.cast<float>();
}
Pose6D GTSPoseToPose6D(const gtsam::Pose3& pose){
  gtsam::Point3 t = pose.translation();
  gtsam::Rot3 R = pose.rotation();
  Pose6D out;
  out.roll = R.roll();
  out.pitch = R.pitch();
  out.yaw = R.yaw();
  out.x = t.x();
  out.y = t.y();
  out.z = t.z();
  return out;
}

void CutVoxel3d(std::unordered_map<VOXEL_LOC, int> &feat_map,
                const pcl::PointCloud<PointType>::Ptr pl_feat, float voxel_box_size){
  uint plsize = pl_feat->size();
  for(uint i=0; i<plsize; i++)
  {
    // Transform point to world coordinate
    PointType &p_c = pl_feat->points[i];
    Eigen::Vector3d pvec_tran(p_c.x, p_c.y, p_c.z);

    // Determine the key of hash table
    float loc_xyz[3];
    for(int j=0; j<3; j++)
    {
      loc_xyz[j] = pvec_tran[j] / voxel_box_size;
      if(loc_xyz[j] < 0)
      {
        loc_xyz[j] -= 1.0;
      }
    }
    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);

    // Find corresponding voxel
    PointType a_pt;
    auto iter = feat_map.find(position);
    if(iter != feat_map.end())
    {

    }
    else // If not finding, build a new voxel
    {
      feat_map[position] = 0;
    }
  }
}

bool CheckIfJustPlane(const pcl::PointCloud<PointType>::Ptr& cloud_in, const float &thr){
  pcl::SACSegmentation<PointType> seg;
  pcl::PointIndices inliners;
  pcl::ModelCoefficients coef;
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setDistanceThreshold(0.3);
  seg.setInputCloud(cloud_in);
  seg.segment(inliners, coef);
  float ratio = float(inliners.indices.size())/float(cloud_in->size()) ;
//  opt_debug_file <<  "plane_ratio: " << ratio << std::endl;
  if (ratio > thr)
    return true;
  return false;
}

void DownsampleCloud(pcl::PointCloud<PointType>& cloud_in, const float &leafsize){
  if (cloud_in.empty())return;
  if (leafsize < 0.01) return;
  pcl::PointCloud<PointType> cloud_ds;
  pcl::PointCloud<PointType>::Ptr cloud_tmp(new pcl::PointCloud<PointType>());
  cloud_tmp->points = cloud_in.points;
  cloud_tmp->height = cloud_in.height;
  cloud_tmp->width  = cloud_in.width ;
  cloud_tmp->header = cloud_in.header;
  pcl::VoxelGrid<PointType> sor;
  sor.setInputCloud(cloud_tmp);
  sor.setLeafSize(leafsize, leafsize, leafsize);
  sor.filter(cloud_ds);
  cloud_in = cloud_ds;
}

bool ICP_Refine(const pcl::PointCloud<PointType>::Ptr& cureKeyframeCloud, const pcl::PointCloud<PointType>::Ptr& targetKeyframeCloud,
         const Eigen::Matrix4f& T_cur, const Eigen::Matrix4f& T_prev, const Eigen::Matrix4f& guess, gtsam::Pose3 &pose_out, float &score,
                pcl::PointCloud<PointType>::Ptr& result){
//  opt_debug_file << "ICP_Refine for " << lc_prev_idx << " , " << lc_curr_idx << std::endl;

  pcl::PointCloud<PointType>::Ptr cloud_cur(new pcl::PointCloud<PointType>());
  pcl::IterativeClosestPoint<PointType, PointType> icp;
  icp.setMaxCorrespondenceDistance(250);
  icp.setMaximumIterations(20);
  icp.setTransformationEpsilon(1e-6);
  icp.setEuclideanFitnessEpsilon(1e-6);
  icp.setRANSACIterations(0);

  pcl::PointCloud<PointType>::Ptr cureKeyframeCloud_centred(new pcl::PointCloud<PointType>());
  pcl::PointCloud<PointType>::Ptr targetKeyframeCloud_centred(new pcl::PointCloud<PointType>());

  pcl::transformPointCloud(*cureKeyframeCloud, *cureKeyframeCloud_centred, T_cur.inverse());
  pcl::transformPointCloud(*targetKeyframeCloud, *targetKeyframeCloud_centred, T_prev.inverse());

  // Align pointclouds
  icp.setInputSource(cureKeyframeCloud_centred);
  icp.setInputTarget(targetKeyframeCloud_centred);
//  pcl::PointCloud<PointType>::Ptr result(new pcl::PointCloud<PointType>());
  icp.align(*result, guess);
  float loopFitnessScoreThreshold = 2; // user parameter but fixed low value is safe.
  score = icp.getFitnessScore();
//  icp_score = score;
//#ifdef save_pcd
//  pcl::transformPointCloud(*cureKeyframeCloud_centred, *cloud_cur, guess);
//  pcl::io::savePCDFileBinary(pgo_scan_directory + "cloud_cur_guess.pcd", *cloud_cur); // scan
//  pcl::io::savePCDFileBinary(pgo_scan_directory + "/ICP/" + std::to_string(lc_prev_idx) + "_" + std::to_string(lc_curr_idx) + "cloud_current.pcd", *cloud_cur); // scan
//  pcl::io::savePCDFileBinary(pgo_scan_directory + "/ICP/" + std::to_string(lc_prev_idx) + "_" + std::to_string(lc_curr_idx) + "_" + std::to_string(score) + "_result.pcd", *result); // scan
//  pcl::io::savePCDFileBinary(pgo_scan_directory + "/ICP/" + std::to_string(lc_prev_idx) + "_" + std::to_string(lc_curr_idx) + "cloud_target.pcd", *targetKeyframeCloud_centred); // scan
//#endif

  if (icp.hasConverged() == false || icp.getFitnessScore() > loopFitnessScoreThreshold) {
//    opt_debug_file << "ICP fitness test failed (" << icp.getFitnessScore() << " > " << loopFitnessScoreThreshold << std::endl;
    return false;
  } else {
//    opt_debug_file << "ICP fitness test passed (" << icp.getFitnessScore() << " < " << loopFitnessScoreThreshold << std::endl;
    // Get pose transformation
    float x, y, z, roll, pitch, yaw;
    Eigen::Affine3f correctionLidarFrame;
    correctionLidarFrame = icp.getFinalTransformation();
    pcl::getTranslationAndEulerAngles (correctionLidarFrame, x, y, z, roll, pitch, yaw);
    pose_out = gtsam::Pose3(gtsam::Rot3::RzRyRx(roll, pitch, yaw), gtsam::Point3(x, y, z));
    return true;
  }

}

#endif // COMMON_LIB_HPP
