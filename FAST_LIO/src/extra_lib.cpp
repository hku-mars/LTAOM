#include "extra_lib.h"

namespace ExtraLib {

PointVector id_vec;
void pubCorrectionIds(const ros::Publisher &pub_handle, const V3D &pos, const int id){
  visualization_msgs::MarkerArray text_msg;
  visualization_msgs::Marker a_text;
  a_text.header.stamp = ros::Time::now();
  a_text.header.frame_id = "camera_init";
  a_text.ns = "correction_ids";
  a_text.action = visualization_msgs::Marker::ADD;
  a_text.pose.orientation.w = 1.0;
  a_text.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
  a_text.scale.z = 7;
  PointType apt;
  apt.x = pos[0], apt.y = pos[1], apt.z = pos[2], apt.intensity = id;
  id_vec.push_back(apt);

  for (int i = 0; i < id_vec.size(); i++){
    a_text.id = i;
    a_text.color.g = 1.0f;
    a_text.color.r = 1.0f;
    a_text.color.a = 1.0f;
    geometry_msgs::Pose pose;
    pose.position.x = id_vec[i].x;
    pose.position.y = id_vec[i].y;
    pose.position.z = id_vec[i].z;
    a_text.pose = pose;
    a_text.text = std::to_string(int(id_vec[i].intensity));
    text_msg.markers.push_back(a_text);
  }

  pub_handle.publish(text_msg);
}

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


//PointCloudXYZI::Ptr feats_down_guess(new PointCloudXYZI(100000, 1));
//vector<float> pointSearchSqDis(1);
//PointVector points_near(1);
//void setKFPose(esekfom::esekf<state_ikfom, 12, input_ikfom> & kf, state_ikfom &tmp_state, const boost::shared_ptr<KD_TREE> &ikd_in, \
//               const PointCloudXYZI::Ptr &feats_down_body, const MD(4,4) &Tcomb, const MD(4,4) &Tcomb_nooff, const int pos_id_lc,\
//               std_msgs::Float32MultiArrayPtr &notification_msg, std_msgs::Float64MultiArrayPtr &notification_msg2, std::ostream &fout_dbg){
//  fout_dbg<<"--------------------setKFPose------------------------------" <<endl;

//  fout_dbg << "Tcomb_nooff: " << Tcomb_nooff << endl;

//  notification_msg->data.push_back(2);
//  notification_msg->data.push_back(RotMtoEuler(tmp_state.rot.toRotationMatrix())[0]);
//  notification_msg->data.push_back(RotMtoEuler(tmp_state.rot.toRotationMatrix())[1]);
//  notification_msg->data.push_back(RotMtoEuler(tmp_state.rot.toRotationMatrix())[2]);
//  notification_msg->data.push_back(tmp_state.pos[0]);
//  notification_msg->data.push_back(tmp_state.pos[1]);
//  notification_msg->data.push_back(tmp_state.pos[2]);
//  notification_msg2->data.push_back(2);
//  notification_msg2->data.push_back(RotMtoEuler(tmp_state.rot.toRotationMatrix())[0]);
//  notification_msg2->data.push_back(RotMtoEuler(tmp_state.rot.toRotationMatrix())[1]);
//  notification_msg2->data.push_back(RotMtoEuler(tmp_state.rot.toRotationMatrix())[2]);
//  notification_msg2->data.push_back(tmp_state.pos[0]);
//  notification_msg2->data.push_back(tmp_state.pos[1]);
//  notification_msg2->data.push_back(tmp_state.pos[2]);
//  fout_dbg << "kf pos " << kf.get_x().pos.transpose() << endl;

//  tmp_state.rot = Tcomb_nooff.block<3,3>(0,0);
//  tmp_state.pos = Tcomb_nooff.block<3,1>(0,3);
//  kf.change_x(tmp_state);

//  notification_msg->data.push_back(RotMtoEuler(tmp_state.rot.toRotationMatrix())[0]);
//  notification_msg->data.push_back(RotMtoEuler(tmp_state.rot.toRotationMatrix())[1]);
//  notification_msg->data.push_back(RotMtoEuler(tmp_state.rot.toRotationMatrix())[2]);
//  notification_msg->data.push_back(tmp_state.pos[0]);
//  notification_msg->data.push_back(tmp_state.pos[1]);
//  notification_msg->data.push_back(tmp_state.pos[2]);
//  notification_msg2->data.push_back(RotMtoEuler(tmp_state.rot.toRotationMatrix())[0]);
//  notification_msg2->data.push_back(RotMtoEuler(tmp_state.rot.toRotationMatrix())[1]);
//  notification_msg2->data.push_back(RotMtoEuler(tmp_state.rot.toRotationMatrix())[2]);
//  notification_msg2->data.push_back(tmp_state.pos[0]);
//  notification_msg2->data.push_back(tmp_state.pos[1]);
//  notification_msg2->data.push_back(tmp_state.pos[2]);
//  fout_dbg << "kf pos " << kf.get_x().pos.transpose() << endl;

//  int cloud_size = feats_down_body->size();
//  assert(cloud_size < 100000);
////  pcl::transformPointCloud(*feats_down_body, *feats_down_guess, Tcomb);
//  M3D Rguess = Tcomb.block<3,3>(0,0);
//  V3D tguess = Tcomb.block<3,1>(0,3);
//  float rmse = 0.0f;
//  for (int i = 0; i < cloud_size; i++){
//    const PointType point_body = feats_down_body->points[i];
//    PointType &point_world = feats_down_guess->points[i];
//    V3D p_body(point_body.x, point_body.y, point_body.z);
//    V3D p_global(Rguesw*p_body + tguess);
//    point_world.x = p_global(0);
//    point_world.y = p_global(1);
//    point_world.z = p_global(2);
//    point_world.intensity = point_body.intensity;

//    ikd_in->Nearest_Search(point_world, 1, points_near, pointSearchSqDis);
//    rmse += sqrt(pointSearchSqDis[0]);
//  }
//  fout_dbg<<endl<<"setpose rmse :"<<rmse/cloud_size<<endl;
//}

V3D esti_center(const PointVector &point_near){
  V3D out(0,0,0);
  for(auto apt:point_near)
    out += V3D(apt.x, apt.y, apt.z);

  out /= point_near.size();
  return out;
}

void eigenRtToPoseMsg(const M3D &R, const V3D &t, geometry_msgs::Pose &out){
  out.position.x = t[0];
  out.position.y = t[1];
  out.position.z = t[2];
  Eigen::Quaterniond quat = Eigen::Quaterniond(R);
  out.orientation.x = quat.x();
  out.orientation.y = quat.y();
  out.orientation.z = quat.z();
  out.orientation.w = quat.w();
}

M3D eulToRotM(double roll, double pitch, double yaw){
  double cx = cos(roll);
  double sx = sin(roll);
  double cy = cos(pitch);
  double sy = sin(pitch);
  double cz = cos(yaw);
  double sz = sin(yaw);
  M3D R;
  R <<           cy*cz,           -cy*sz,     sy,
       cx*sz + cz*sx*sy, cx*cz - sx*sy*sz, -cy*sx,
       sx*sz - cx*cz*sy, cz*sx + cx*sy*sz,  cx*cy;
  return R;
}

M3D quatToRotM(double w, double x, double y, double z){
  M3D R;
  R <<  1 - 2*(y*y + z*z),   2*(x*y - w*z),   2*(x*z + w*y),
      2*(x*y + w*z), 1 - 2*(x*x + z*z),   2*(y*z - w*x),
      2*(x*z - w*y),   2*(y*z + w*x), 1 - 2*(x*x + y*y);
  return R;
}

void poseMsgToEigenRT(const geometry_msgs::Pose &m, M3D &R, V3D &t){
  t = V3D(m.position.x, m.position.y, m.position.z);
  Eigen::Quaterniond q(m.orientation.w, m.orientation.x,\
                       m.orientation.y, m.orientation.z);
  R = q.toRotationMatrix();
//  R = quatToRotM(m.orientation.w, m.orientation.x, m.orientation.y, m.orientation.z);
}

V3D geometryOrientationToRPY(const geometry_msgs::Quaternion pose_in){
  tf::Quaternion q_tmp(pose_in.x, pose_in.y, \
                       pose_in.z, pose_in.w);
  double roll, pitch, yaw;
  tf::Matrix3x3(q_tmp).getRPY(roll, pitch, yaw);
  return V3D(roll, pitch, yaw);
}


void printInfo(std::unordered_map<int, SubmapInfo> &unmap_submap_info, ostream& fout_dbg){
  fout_dbg<< "unmap_submap_info.size(): " << unmap_submap_info.size() << endl;
  for (int i = 0; i < unmap_submap_info.size(); i++){
    auto iter = unmap_submap_info.find(i);
    if (iter != unmap_submap_info.end())
    fout_dbg<<iter->second.submap_index<<" cloud.size() :"<<iter->second.cloud_ontree.size() \
           <<" ori_set_posed :"<<iter->second.oriPoseSet<<" cor_set_posed :"<<iter->second.corPoseSet<<endl;
  }
  fout_dbg<<"-------------------------------------" <<endl;
}

PointVector findSubmapBoundingBoxPt(const PointVector &submap_cloud){
  PointVector result;
  if (submap_cloud.empty()) return result;
  float x_max = -std::numeric_limits<float>::infinity();
  float y_max = x_max;
  float z_max = x_max;
  float x_min = -x_max;
  float y_min = x_min;
  float z_min = x_min;
  for (auto &a_pt:submap_cloud){
    float x = a_pt.x; float y = a_pt.y; float z = a_pt.z;
    x_min = x_min>x?x:x_min; y_min = y_min>y?y:y_min;  z_min = z_min>z?z:z_min;
    x_max = x_max<x?x:x_max; y_max = y_max<y?y:y_max;  z_max = z_max<z?z:z_max;
  }
  PointType pt_max, pt_min;
  pt_max.x = x_max; pt_max.y = y_max; pt_max.z = z_max;
  pt_min.x = x_min; pt_min.y = y_min; pt_min.z = z_min;
  result.push_back(pt_max);
  result.push_back(pt_min);
  return result;
}

float calcScanSubmapOverlapRatio(const PointType scan_pt_max, const PointType scan_pt_min, const PointType submap_pt_max, const PointType submap_pt_min){
  if (scan_pt_max.x < submap_pt_min.x || scan_pt_min.x > submap_pt_max.x) return 0.0f;
  if (scan_pt_max.y < submap_pt_min.y || scan_pt_min.y > submap_pt_max.y) return 0.0f;
  if (scan_pt_max.z < submap_pt_min.z || scan_pt_min.z > submap_pt_max.z) return 0.0f;
  float x_span, y_span, z_span;
  if (scan_pt_max.x < submap_pt_max.x && scan_pt_min.x > submap_pt_min.x)
    x_span = scan_pt_max.x - scan_pt_min.x;
  else
    x_span = fabs(scan_pt_max.x - submap_pt_min.x)>fabs(scan_pt_min.x - submap_pt_max.x)?\
        fabs(scan_pt_min.x - submap_pt_max.x):fabs(scan_pt_max.x - submap_pt_min.x);

  if (scan_pt_max.y < submap_pt_max.y && scan_pt_min.y > submap_pt_min.y)
    y_span = scan_pt_max.y - scan_pt_min.y;
  else
    y_span = fabs(scan_pt_max.y - submap_pt_min.y)>fabs(scan_pt_min.y - submap_pt_max.y)?\
        fabs(scan_pt_min.y - submap_pt_max.y):fabs(scan_pt_max.y - submap_pt_min.y);

  if (scan_pt_max.z < submap_pt_max.z && scan_pt_min.z > submap_pt_min.z)
    z_span = scan_pt_max.z - scan_pt_min.z;
  else
    z_span = fabs(scan_pt_max.z - submap_pt_min.z)>fabs(scan_pt_min.z - submap_pt_max.z)?\
        fabs(scan_pt_min.z - submap_pt_max.z):fabs(scan_pt_max.z - submap_pt_min.z);

  float overlap_area = x_span*y_span*z_span;
  float scan_area = (scan_pt_max.x - scan_pt_min.x)*(scan_pt_max.y - scan_pt_min.y)*(scan_pt_max.z - scan_pt_min.z);
  float ratio = overlap_area/scan_area;
  return ratio;
}


void CutVoxel3d(std::unordered_map<VOXEL_LOC, OCTO_TREE_NEW*> &feat_map,
                const pcl::PointCloud<PointType>::Ptr  pl_feat, float voxel_box_size){
  uint plsize = pl_feat->size();
  for(uint i=0; i<plsize; i++)
  {
    // Transform point to world coordinate
    PointType p_c = pl_feat->points[i];
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
      //      iter->second->plvec_orig[fnum]->push_back(pvec_orig);
      iter->second->is2opt = true;
      a_pt.x = pvec_tran[0], a_pt.y = pvec_tran[1], a_pt.z = pvec_tran[2];
      iter->second->plvec_pcl->points.push_back(a_pt);
    }
    else // If not finding, build a new voxel
    {
      OCTO_TREE_NEW *ot = new OCTO_TREE_NEW();
      //      ot->plvec_orig[fnum]->push_back(pvec_orig);
      a_pt.x = pvec_tran[0], a_pt.y = pvec_tran[1], a_pt.z = pvec_tran[2];
      ot->plvec_pcl->points.push_back(a_pt);

      // Voxel center coordinate
      ot->voxel_center[0] = (0.5+position.x) * voxel_box_size;
      ot->voxel_center[1] = (0.5+position.y) * voxel_box_size;
      ot->voxel_center[2] = (0.5+position.z) * voxel_box_size;
      ot->quater_length = voxel_box_size / 4.0; // A quater of side length
      feat_map[position] = ot;
    }
  }
}

cv::Mat CreateImage(std::unordered_map<VOXEL_LOC, OCTO_TREE_NEW*> &voxels_uomap_flat,\
                    std::unordered_map<VOXEL_LOC, OCTO_TREE_NEW*> &voxels_uomap_flat2,\
                    const float &v_size, const bool &to_print){

  double x_max = -std::numeric_limits<double>::infinity();
  double y_max = x_max;
  double x_min = -x_max;
  double y_min = x_min;
  std::vector<Eigen::Vector2f> voxel_centers;
//  fout_dbg << "voxels_uomap_flat.size(): " << voxels_uomap_flat.size() << std::endl;

  for(auto iter=voxels_uomap_flat.begin(); iter!=voxels_uomap_flat.end(); ++iter)
  {
    float x = iter->second->voxel_center[0];
    float y = iter->second->voxel_center[1];

    voxel_centers.push_back(Eigen::Vector2f(x, y));
    x_min = x_min>x?x:x_min;
    y_min = y_min>y?y:y_min;
    x_max = x_max<x?x:x_max;
    y_max = y_max<y?y:y_max;
  }
//  fout_dbg << "voxel_centers.size(): " << voxel_centers.size() << std::endl;

  int padding = 4;
  int image_width = int(roundf((x_max - x_min)/v_size)) + 2*padding;
  int image_height = int(roundf((y_max - y_min)/v_size)) + 2*padding;
//  fout_dbg << x_max << x_min << std::endl;
//  fout_dbg << y_max << y_min << std::endl;
//  fout_dbg << "image_width: " << image_width << std::endl;
//  fout_dbg << "image_height:" << image_height << std::endl;

  cv::Mat result_img = cv::Mat::zeros(image_height, image_width, CV_8UC1);
  for(int i = 0; i < voxel_centers.size(); i++)
  {
    int r = int(roundf((voxel_centers[i][1]-y_min)/v_size)) + padding;
    int c = int(roundf((voxel_centers[i][0]-x_min)/v_size)) + padding;
    result_img.at<uchar>(r,c) = 100;
  }

//  double x_max2 = -std::numeric_limits<double>::infinity();
//  double y_max2 = x_max2;
//  double x_min2 = -x_max2;
//  double y_min2 = x_min2;
  std::vector<Eigen::Vector2f> voxel_centers2;
//  fout_dbg << "voxels_uomap_flat2.size(): " << voxels_uomap_flat2.size() << std::endl;

  for(auto iter=voxels_uomap_flat2.begin(); iter!=voxels_uomap_flat2.end(); ++iter)
  {
    float x = iter->second->voxel_center[0];
    float y = iter->second->voxel_center[1];

    if(x_min>x) continue;
    if(y_min>y) continue;
    if(x_max<x) continue;
    if(y_max<y) continue;

    voxel_centers2.push_back(Eigen::Vector2f(x, y));
  }
  std::cout << "voxel_centers2.size(): " << voxel_centers2.size() << std::endl;

//  fout_dbg << x_max2 << x_min2 << std::endl;
//  fout_dbg << y_max2 << y_min2 << std::endl;
//  assert(x_min<=x_min2);
//  assert(y_min<=y_min2);
//  assert(x_max>=x_max2);
//  assert(y_max>=y_max2);

  for(int i = 0; i < voxel_centers2.size(); i++)
  {
    int r = int(roundf((voxel_centers2[i][1]-y_min)/v_size)) + padding;
    int c = int(roundf((voxel_centers2[i][0]-x_min)/v_size)) + padding;
    result_img.at<uchar>(r,c) = 255;
  }

//  if (to_print)
//    fout_dbg << "result_img" << std::endl;
//  for(int r = 0; r < image_height; r++)
//  {
//    for(int c = 0; c < image_width; c++)
//    {
//      if (result_img.at<uchar>(r, c) > 0){
//        auto val = result_img.at<uchar>(r,c);
//        if (to_print)
//          fout_dbg << " " << int(val);
//      }
//      else{
//        if (to_print)
//          fout_dbg << " " << 0;
//      }
//    }
//    if (to_print)
//      fout_dbg << std::endl;
//  }

  return result_img;
}

}

