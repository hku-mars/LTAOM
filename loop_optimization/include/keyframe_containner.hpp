#ifndef KEYFRAME_CONTAINNER_HPP
#define KEYFRAME_CONTAINNER_HPP

#include "predefined_types.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <ros/ros.h>

#include <vector>

class KeyFrame
{
public:
  KeyFrame() {
    pose_opt_set = false;
    this->KeyCloud.reset(new pcl::PointCloud<PointType>());
  }
  ~KeyFrame() {}

  Pose6D KeyPose;
  Pose6D KeyPoseOpt;
  Pose6D KeyPoseCompare;
  bool pose_opt_set;
  ros::Time KeyTime;
  pcl::PointCloud<PointType>::Ptr KeyCloud;

private:

};


#endif // KEYFRAME_CONTAINNER_HPP
