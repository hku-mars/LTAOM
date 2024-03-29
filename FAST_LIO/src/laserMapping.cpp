/*This code is the implementation of our paper "LTA-OM: Long-Term Association
 LiDAR-Inertial Odometry and Mapping".

Current Developer: Zuhao Zou < zuhaozou@yahoo.com >

If you use any code of this repo in your academic research, please cite at least
one of our papers:
[1] Zou, Zuhao, et al. "LTA-OM: Long-Term Association LiDAR-Inertial
    Odometry and Mapping"
[2] Yuan, C., et al. "Std: Stable triangle descriptor for 3d place recognition"
[3] Xu, Wei, et al. "Fast-lio2: Fast direct lidar-inertial odometry."
[4] Xu, Wei, and Fu Zhang. "Fast-lio: A fast, robust lidar-inertial odometry
    package by tightly-coupled iterated kalman filter."
[5] Cai, Yixi, Wei Xu, and Fu Zhang. "ikd-Tree: An Incremental KD Tree for
    Robotic Applications."
[6] Lin, Jiarong, and Fu Zhang. "Loam-livox: A fast, robust, high-precision
    LiDAR odometry and mapping package for LiDARs of small FoV."

For commercial use, please contact me < zuhaozou@yahoo.com > and
Dr. Fu Zhang < fuzhang@hku.hk >.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.
 3. Neither the name of the copyright holder nor the names of its
    contributors may be used to endorse or promote products derived from this
    software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 POSSIBILITY OF SUCH DAMAGE.
*/
#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <ros/ros.h>
#include <Eigen/Core>
// #include <common_lib.h>-
#include "IMU_Processing.hpp"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <fast_lio/States.h>
#include <geometry_msgs/Vector3.h>
#include <livox_ros_driver/CustomMsg.h>
#include "preprocess.h"

#ifdef USE_ikdforest
#include <ikd-Forest/ikd_Forest.h>
#else
#include <ikd-Tree/ikd_Tree.h>
#endif

#ifndef DEPLOY
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;
#endif

#include "extra_lib.h"
#include <std_msgs/Int32.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/UInt64.h>
//#include <unordered_map>
//#include <visualization_msgs/MarkerArray.h>

//#define offline
//#define LoadBag
#ifdef LoadBag
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <tf/transform_broadcaster.h>
//#define nclt
//#define avia_indoor
#endif

#define as_node
#ifdef as_node
#include "laserMapping.h"
#endif

#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)
#define MAXN                (720000)
#define PUBFRAME_PERIOD     (20)

float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;

mutex mtx_buffer;
condition_variable sig_buffer;

string root_dir = ROOT_DIR;
string map_file_path, lid_topic, imu_topic;
double time_offset_from_lidar = 0.0f;
// int iterCount, feats_down_size, NUM_MAX_ITERATIONS, laserCloudValidNum,\
//     effct_feat_num, time_log_counter, publish_count = 0;

int iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0,\
    effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;

double res_mean_last = 0.05;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;


// Time Log Variables
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
int kdtree_delete_counter = 0;
int kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0;
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];
double search_time_rec[100000];

double match_time = 0, solve_time = 0, solve_const_H_time = 0;

bool lidar_pushed, flg_reset, flg_exit = false, flg_EKF_inited;
bool dense_map_en = true;

vector<BoxPointType> cub_needrm;

// deque<sensor_msgs::PointCloud2::ConstPtr> lidar_buffer;
deque<PointCloudXYZI::Ptr>  lidar_buffer;
deque<double>          time_buffer;
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;
vector<vector<int>> pointSearchInd_surf;
vector<PointVector> Nearest_Points;
vector<double> extrinT(3, 0.0);
vector<double> extrinR(9, 0.0);
bool   point_selected_surf[100000] = {0};
float  res_last[100000] = {0.0};
double total_residual ;
uint64_t curr_Mea_lidarbegtime_NS = 0;

//#define save_for_mapconsistency_eva  //uncomment to generate dense scan poses and undistorted scans
#ifdef save_for_mapconsistency_eva
#include <rosbag/bag.h>
int cnt_scan = 0;
std::deque<std::tuple<double, Eigen::Matrix3d, Eigen::Vector3d>> scan_poses_queue;
std::deque<std::tuple<double, Eigen::Matrix3d, Eigen::Vector3d>> extrinsics_queue;
std::string scanpose_cor_filename = "";
#endif

//surf feature in map
PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr _featsArray;
pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;

//    KD_TREE ikdtree;
boost::shared_ptr<KD_TREE<PointType>> ikdtree_ptr    (boost::make_shared<KD_TREE<PointType>>());
boost::shared_ptr<KD_TREE<PointType>> ikdtree_swapptr(boost::make_shared<KD_TREE<PointType>>());
std::mutex mtx_ikdtreeptr;
std::condition_variable sig_ikdtreeptr;

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur;
V3D position_last(Zero3d);

//estimator inputs and output;
MeasureGroup Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
state_ikfom state_point;
vect3 pos_lid;

V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

nav_msgs::Path path;
nav_msgs::Odometry odomAftMapped;
geometry_msgs::Quaternion geoQuat;
geometry_msgs::PoseStamped msg_body_pose;
Eigen::Matrix3d rot_replace;
Eigen::Vector3d pos_replace;
Eigen::Vector3d vel_replace;
bool do_posecorrection = false;
shared_ptr<Preprocess> p_pre(new Preprocess());

int multisession_mode = 0; //disabled by default
double correction_ver_thr = 0.45;
double correction_dis_interval = 50; // 50 is a very frequent number
double dy_mapretrival_range = 150;

std::unordered_map<int, SubmapInfo> unmap_submap_info;
deque<nav_msgs::Path::ConstPtr>  path_buffer;
ros::Publisher pubLaserCloudFullCor, pubOdomLargeJump, pubOdomCorrection, pubOdomCorrection2, pubCorrectionId, pubTimeCorrection;
ofstream fout_pre, fout_out, fout_dbg;
std::fstream time_ikdrebuild_thread;
mutex mtx_sub_;
std::deque<sensor_msgs::PointCloud2ConstPtr> submap_buffer;
std::deque<nav_msgs::OdometryConstPtr> submap_pose_buffer;
PointVector PointSubmap;
std::deque<PointCloudXYZI::Ptr>  PointToAddHistorical;
int submap_id;
pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr pos_kdtree, pos_kdtree_prior;
bool holding_for_ikdtreerebuild = false;
bool first_correction_set = false;
std::string save_directory;

PointCloudXYZI::Ptr correctd_cloud_submap(new PointCloudXYZI());
state_ikfom tmp_state;

void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}

inline void dump_lio_state_to_log(FILE *fp)
{
    //state_ikfom write_state = kf.get_x();
    V3D rot_ang(Log(state_point.rot.toRotationMatrix()));
    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                   // Angle
    fprintf(fp, "%lf %lf %lf ", state_point.pos(0), state_point.pos(1), state_point.pos(2)); // Pos
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // omega
    fprintf(fp, "%lf %lf %lf ", state_point.vel(0), state_point.vel(1), state_point.vel(2)); // Vel
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // Acc
    fprintf(fp, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1), state_point.bg(2));    // Bias_g
    fprintf(fp, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1), state_point.ba(2));    // Bias_a
    fprintf(fp, "%lf %lf %lf ", state_point.grav[0], state_point.grav[1], state_point.grav[2]); // Bias_a
    fprintf(fp, "\r\n");
    fflush(fp);
}

//project the lidar scan to world frame
void pointBodyToWorld_ikfom(PointType const * const pi, PointType * const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    //state_ikfom transfer_state = kf.get_x();
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);
    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template<typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    //state_ikfom transfer_state = kf.get_x();
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);
    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    //state_ikfom transfer_state = kf.get_x();
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);
    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
    float intensity = pi->intensity;
    intensity = intensity - floor(intensity);
    int reflection_map = intensity*10000;
}

int points_cache_size = 0;

void points_cache_collect()
{
    PointVector points_history;
    ikdtree_ptr->acquire_removed_points(points_history);
    points_cache_size = points_history.size();
    for (int i = 0; i < points_history.size(); i++) _featsArray->push_back(points_history[i]);
}

BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;
void lasermap_fov_segment()
{
    cub_needrm.clear();
    kdtree_delete_counter = 0;
    kdtree_delete_time = 0.0;
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
    //state_ikfom fov_state = kf.get_x();
    //V3D pos_LiD = fov_state.pos + fov_state.rot * fov_state.offset_T_L_I;
    V3D pos_LiD = pos_lid;
    if (!Localmap_Initialized){
        //if (cube_len <= 2.0 * MOV_THRESHOLD * DET_RANGE) throw std::invalid_argument("[Error]: Local Map Size is too small! Please change parameter \"cube_side_length\" to larger than %d in the launch file.\n");
        for (int i = 0; i < 3; i++){
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }
    // printf("Local Map is (%0.2f,%0.2f) (%0.2f,%0.2f) (%0.2f,%0.2f)\n", LocalMap_Points.vertex_min[0],LocalMap_Points.vertex_max[0],LocalMap_Points.vertex_min[1],LocalMap_Points.vertex_max[1],LocalMap_Points.vertex_min[2],LocalMap_Points.vertex_max[2]);
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++){
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;
    }
    if (!need_move) return;
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));
    for (int i = 0; i < 3; i++)
    {
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE)
        {
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
            // printf("Delete Box is (%0.2f,%0.2f) (%0.2f,%0.2f) (%0.2f,%0.2f)\n", tmp_boxpoints.vertex_min[0],tmp_boxpoints.vertex_max[0],tmp_boxpoints.vertex_min[1],tmp_boxpoints.vertex_max[1],tmp_boxpoints.vertex_min[2],tmp_boxpoints.vertex_max[2]);
        } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
        {
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
            // printf("Delete Box is (%0.2f,%0.2f) (%0.2f,%0.2f) (%0.2f,%0.2f)\n", tmp_boxpoints.vertex_min[0],tmp_boxpoints.vertex_max[0],tmp_boxpoints.vertex_min[1],tmp_boxpoints.vertex_max[1],tmp_boxpoints.vertex_min[2],tmp_boxpoints.vertex_max[2]);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    double delete_begin = omp_get_wtime();
    if(cub_needrm.size() > 0) kdtree_delete_counter = ikdtree_ptr->Delete_Point_Boxes(cub_needrm);
    kdtree_delete_time = omp_get_wtime() - delete_begin;
    // printf("Delete time: %0.6f, delete size: %d\n",kdtree_delete_time,kdtree_delete_counter);
    // printf("Delete Box: %d\n",int(cub_needrm.size()));
}

void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    mtx_buffer.lock();
    scan_count ++;
    double preprocess_start_time = omp_get_wtime();
    double time_offset = 0.0f;
    if (time_offset_from_lidar != 0) time_offset = time_offset_from_lidar;
    if (msg->header.stamp.toSec() + time_offset< last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    //std::cout << "time_offset: " << time_offset << std::endl;
    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    if (p_pre->lidar_type == XGRIDS)
        pcl::fromROSMsg(*msg, *ptr);
    else
        p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(msg->header.stamp.toSec() + time_offset);
    last_timestamp_lidar = msg->header.stamp.toSec() + time_offset;
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    fout_dbg << "last_timestamp_lidar: " << std::to_string(last_timestamp_lidar) << std::endl << std::endl;
}

void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg)
{
    mtx_buffer.lock();
    scan_count ++;
    double preprocess_start_time = omp_get_wtime();
    // cout<<"got feature"<<endl;
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    // ROS_INFO("get point cloud at time: %.6f", msg->header.stamp.toSec());
    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(msg->header.stamp.toSec());
    last_timestamp_lidar = msg->header.stamp.toSec();
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in)
{
    //fout_dbg << "imu timestamp: " << std::to_string(msg_in->header.stamp.toSec()) << std::endl;
    publish_count ++;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    double timestamp = msg->header.stamp.toSec();
    mtx_buffer.lock();

    if (timestamp < last_timestamp_imu)
    {
        ROS_ERROR("imu loop back, clear buffer");
        imu_buffer.clear();
        flg_reset = true;
    }

    last_timestamp_imu = timestamp;

    imu_buffer.push_back(msg);
    // cout<<"got imu: "<<timestamp<<" imu size "<<imu_buffer.size()<<endl;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}
double last_time = 0;
bool sync_packages(MeasureGroup &meas)
{
    if (lidar_buffer.empty() || imu_buffer.empty())
    {
        return false;
    }

    /*** push a lidar scan ***/
    if(!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();
        if(meas.lidar->points.size() <= 1)
        {
            lidar_buffer.pop_front();
            return false;
        }
        if (p_pre->lidar_type == XGRIDS)
        {
            meas.lidar_beg_time = time_buffer.front() ;
            lidar_end_time = meas.lidar_beg_time + 0.1;
        }
        else
        {
            meas.lidar_beg_time = time_buffer.front();
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
        }
        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = imu_buffer.front()->header.stamp.toSec();
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = imu_buffer.front()->header.stamp.toSec();
        if(imu_time > lidar_end_time + 0.02) break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    last_time = omp_get_wtime();
    return true;
}

#ifdef LoadBag
void msg_callbacks(){
#ifdef nclt
  std::string bag_path = "/home/zuhaozou/bag_local/nclt/vel_xu_0526.bag";
#else

#ifdef avia_indoor
  std::string bag_path = "/media/zuhaozou/zuhaozou_ssd/rosbag/MarS/Multi-floor building/floor7.bag";
#else
//  std::string bag_path = "/home/zuhaozou/bag_local/mulran/DCC02_raw.bag";
//    std::string bag_path = "/home/zuhaozou/bag_local/mulran/KAIST03_raw.bag";
  std::string bag_path = "/home/zuhaozou/bag_local/mulran/river02_raw.bag";
#endif

#endif
  std::fstream file_;
  file_.open(bag_path, ios::in);
  if (!file_) {
    fout_dbg << "File " << bag_path << " does not exit" << endl;
  }
  ROS_INFO("Start to load the rosbag %s", bag_path.c_str());
  rosbag::Bag bag;
  try {
    bag.open(bag_path, rosbag::bagmode::Read);
  } catch (rosbag::BagException e) {
    ROS_ERROR_STREAM("LOADING BAG FAILED: " << e.what());
  }
  std::vector<string> types;
  types.push_back(string("sensor_msgs/PointCloud2"));
  types.push_back(string("sensor_msgs/Imu"));
  rosbag::View view(bag, rosbag::TypeQuery(types));
  ros::Rate rate(200);
//  int counter = 0;
//  while (ros::ok() ){
//    rate.sleep();
//    counter++;
    BOOST_FOREACH (rosbag::MessageInstance const m, view) {
      rate.sleep();
//      counter++;
//      fout_dbg << "counter: " << counter << std::endl;
      sensor_msgs::PointCloud2::ConstPtr cloud_ptr =
          m.instantiate<sensor_msgs::PointCloud2>();
      if (cloud_ptr != NULL) {
//        fout_dbg << "standard_pcl_cbk" << std::endl;
        standard_pcl_cbk(cloud_ptr);
      }
      sensor_msgs::Imu::ConstPtr imu_ptr = m.instantiate<sensor_msgs::Imu>();
      if (imu_ptr != NULL) {
//        fout_dbg << "imu_cbk" << std::endl;
        imu_cbk(imu_ptr);
      }
//      if (keyboard_interupt){
//        ROS_INFO("Please enter any key to continue!");
////        std::cin.get();
//        keyboard_interupt = false;
//      }
    }

//  }
}
#endif

int process_increments = 0;
void map_incremental()
{
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointVector PointToAddHistoricalFront;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    PointToAddHistoricalFront.reserve(100000);
    double filter_size_map_mi_d = double(filter_size_map_min);
    for (int i = 0; i < feats_down_size; i++)
    {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        /* decide if need add to map */
        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType downsample_result, mid_point;

            double mx, my, mz;
            if (fabs(feats_down_world->points[i].x-round(feats_down_world->points[i].x))<0.0001) feats_down_world->points[i].x = round(feats_down_world->points[i].x)+0.001;
            if (fabs(feats_down_world->points[i].y-round(feats_down_world->points[i].y))<0.0001) feats_down_world->points[i].y = round(feats_down_world->points[i].y)+0.001;
            if (fabs(feats_down_world->points[i].z-round(feats_down_world->points[i].z))<0.0001) feats_down_world->points[i].z = round(feats_down_world->points[i].z)+0.001;
            mx = floor(double(feats_down_world->points[i].x)*2)*filter_size_map_mi_d + 0.5 * filter_size_map_mi_d;
            my = floor(double(feats_down_world->points[i].y)*2)*filter_size_map_mi_d + 0.5 * filter_size_map_mi_d;
            mz = floor(double(feats_down_world->points[i].z)*2)*filter_size_map_mi_d + 0.5 * filter_size_map_mi_d;
            double dist  = calc_dist_ondouble(feats_down_world->points[i], mx, my, mz);
            if (fabs(double(points_near[0].x) - mx) - double(0.5 * filter_size_map_mi_d) > 0 && \
                fabs(double(points_near[0].y) - my) - double(0.5 * filter_size_map_mi_d) > 0 && \
                fabs(double(points_near[0].z) - mz) - double(0.5 * filter_size_map_mi_d) > 0)
            {
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i ++)
            {
                if (points_near.size() < NUM_MATCH_POINTS) break;
                double dist_near = calc_dist_ondouble(points_near[readd_i],  mx, my, mz);
                if (dist_near - dist < 0)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add) PointToAdd.push_back(feats_down_world->points[i]);
            if (need_add && (PointToAdd.size()%3) == 0) PointSubmap.push_back(feats_down_world->points[i]);
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    double st_time = omp_get_wtime();
    if (!PointToAddHistorical.empty())
    {
        PointToAddHistoricalFront = PointToAddHistorical.front()->points;
        PointToAddHistorical.pop_front();
    }
    {
        unique_lock<std::mutex> my_unique_lock(mtx_ikdtreeptr);
        sig_ikdtreeptr.wait(my_unique_lock, []{return !holding_for_ikdtreerebuild;});
        add_point_size = ikdtree_ptr->Add_Points(PointToAdd, true);
        ikdtree_ptr->Add_Points(PointNoNeedDownsample, false);
        if (!PointToAddHistoricalFront.empty())
        {
            add_point_size = ikdtree_ptr->Add_Points(PointToAddHistoricalFront, true);
        }
    }
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
    kdtree_incremental_time = omp_get_wtime() - st_time;
}


template<typename T>
void set_posestamp(T & out)
{
    #ifdef USE_IKFOM
    //state_ikfom stamp_state = kf.get_x();
    out.position.x = state_point.pos(0);
    out.position.y = state_point.pos(1);
    out.position.z = state_point.pos(2);
    #else
    out.position.x = state.pos_end(0);
    out.position.y = state.pos_end(1);
    out.position.z = state.pos_end(2);
    #endif
    out.orientation.x = geoQuat.x;
    out.orientation.y = geoQuat.y;
    out.orientation.z = geoQuat.z;
    out.orientation.w = geoQuat.w;
}

PointCloudXYZI::Ptr feats_down_world_guess(new PointCloudXYZI(100000, 1));
vector<PointVector> PointsNearVec(100000);
float point_dis[100000] = {0.0};
PointCloudXYZI::Ptr point_world_tmp(new PointCloudXYZI());
PointCloudXYZI::Ptr point_near_tmp(new PointCloudXYZI());
double match_time_avg = 0.0f;
int match_count = 0;
bool refine_with_pt2pt_icp(const PointCloudXYZI::Ptr &feats_down_body, const float rmse_thr, const int iteration_thr, const boost::shared_ptr<KD_TREE<PointType>> &ikd_in, \
                        M3D &Ricp, V3D &ticp,  const M3D &Rguess, const V3D &tguess, const int pos_id_lc, std::ostream &fout_dbg);

bool refine_with_pt2_plane_icp(const PointCloudXYZI::Ptr &feats_down_body, const float rmse_thr, const int iteration_thr, const boost::shared_ptr<KD_TREE<PointType>> &ikd_in, \
                           M3D &Ricp, V3D &ticp, const M3D &Rguess, const V3D &tguess, const int pos_id_lc, std::ostream &fout_dbg);


PointCloudXYZI::Ptr feats_down_guess(new PointCloudXYZI(100000, 1));
vector<float> pointSearchSqDis(1);
PointVector points_near(1);
void set_KF_pose(esekfom::esekf<state_ikfom, 12, input_ikfom> & kf, state_ikfom &tmp_state, const boost::shared_ptr<KD_TREE<PointType>> &ikd_in, \
               const PointCloudXYZI::Ptr &feats_down_body, const MD(4,4) &Tcomb, const MD(4,4) &Ticp, const MD(4,4) &Tcomb_nooff, const M3D &Rvel, const int pos_id_lc,\
               std_msgs::Float32MultiArrayPtr &notification_msg, std_msgs::Float64MultiArrayPtr &notification_msg2, std::ostream &fout_dbg);

void correctLidarPoints(PointType const * const pi, PointType * const po, const M3D &rotM, const V3D &tran)
{
    V3D p_ori(pi->x, pi->y, pi->z);
    V3D p_corrected(rotM*p_ori + tran);
    po->x = p_corrected(0);
    po->y = p_corrected(1);
    po->z = p_corrected(2);
    po->intensity = pi->intensity;
}
void correctLidarPoints(PointType const * const pi, PointType * const po, const M3D &R1, const V3D &t1,\
                        const M3D &R2, const V3D &t2)
{
    V3D p_ori(pi->x, pi->y, pi->z);
    V3D p_corrected(R2*R1.transpose()*(p_ori - t1)+t2);
    po->x = p_corrected(0);
    po->y = p_corrected(1);
    po->z = p_corrected(2);
    po->intensity = pi->intensity;
}

MD(4,4) M3D_to_M4D(M3D Min, V3D tin)
{
    MD(4,4) out = MD(4,4)::Identity();
    out.block<3,3>(0,0) = Min;
    out.block<3,1>(0,3) = tin;
    return out;
}

void path_cor_cbk(const nav_msgs::Path::ConstPtr &path_in)
{
    //mtx_sub_.lock();
    path_buffer.push_back(path_in);
    //fout_dbg << "path_in->poses.back().header.stamp.toSec(): " << std::to_string(path_in->poses.back().header.stamp.toSec()) << std::endl;
    //mtx_sub_.unlock();
}

PointCloudXYZI::Ptr last_correction_scan(new PointCloudXYZI());
int last_submap_id;
void submap_id_cbk(const std_msgs::Int32ConstPtr &id_msg)
{
    last_submap_id = submap_id;
    *last_correction_scan = *feats_down_world;
    SubmapInfo tmp;
    submap_id = id_msg->data;
    tmp.submap_index = submap_id;
    tmp.cloud_ontree = PointSubmap;
    fout_dbg << "Submap " << submap_id << "-th has " << PointSubmap.size() <<  " points " << endl;
    PointVector ().swap(PointSubmap);
    auto iter = unmap_submap_info.find(tmp.submap_index);
    if (iter == unmap_submap_info.end())
        unmap_submap_info[tmp.submap_index] = tmp;
}

void submap_pose_cbk(const nav_msgs::Odometry::ConstPtr &odom_msg)
{
    mtx_sub_.lock();
    submap_pose_buffer.push_back(odom_msg);
    mtx_sub_.unlock();
}

bool update_submap_info()
{
    if (submap_pose_buffer.empty()) return false;
    if (submap_pose_buffer.back()->twist.covariance[0]!=submap_id) return false;
    mtx_sub_.lock();
    auto submap_pose_buffer_tmp = submap_pose_buffer;
    submap_pose_buffer.clear();
    for (auto &submap_pose : submap_pose_buffer_tmp)
    {
        int idx = int(submap_pose->twist.covariance[0]);
        auto iter = unmap_submap_info.find(idx);
        if (iter != unmap_submap_info.end() && !iter->second.oriPoseSet)
        {
            auto &submap_info = iter->second;
            ExtraLib::poseMsgToEigenRT(submap_pose->pose.pose, submap_info.lidar_pose_rotM, submap_info.lidar_pose_tran);
            submap_info.oriPoseSet = true;
            submap_info.msg_time = submap_pose->header.stamp.toSec();
#ifdef save_for_mapconsistency_eva
            fout_dbg << "submap_pose->twist.covariance[1] " << std::to_string(submap_pose->twist.covariance[1]) << std::endl;
            fout_dbg << "submap_pose->twist.covariance[2] " << std::to_string(submap_pose->twist.covariance[2]) << std::endl;
            for ( ; cnt_scan < scan_poses_queue.size(); )
            {
                cnt_scan++;
                if (std::get<0>(scan_poses_queue[cnt_scan]) < submap_pose->twist.covariance[1])
                    continue;
                if (std::get<0>(scan_poses_queue[cnt_scan]) > submap_pose->twist.covariance[2])
                    continue;
                Eigen::Vector3d t_dif = std::get<2>(scan_poses_queue[cnt_scan]) - std::get<2>(scan_poses_queue[cnt_scan-1]);
                if (t_dif.norm()/fabs(std::get<0>(scan_poses_queue[cnt_scan]) - std::get<0>(scan_poses_queue[cnt_scan-1])) > 50) //50m/s is not possible
                    continue;
                submap_info.scan_poses_insubmap.push_back(scan_poses_queue[cnt_scan]);
                submap_info.extrinsics_insubmap.push_back(extrinsics_queue[cnt_scan]);
            }
#endif
        }
        else
            submap_pose_buffer.push_back(submap_pose);
    }
    mtx_sub_.unlock();

    return true;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr key_poses(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr key_poses_prior(new pcl::PointCloud<pcl::PointXYZI>());
std::unordered_map<int, SubmapInfo> unmap_submap_info_bkq;
#ifdef save_for_mapconsistency_eva
void set_submap_corrected_poses(const nav_msgs::Path::ConstPtr& path_cor, std::fstream &scan_pose_cor_file)
#else
void set_submap_corrected_poses(const nav_msgs::Path::ConstPtr& path_cor)
#endif
{
    if (!key_poses->empty()) key_poses->clear();
    fout_dbg<< "--------------------set_submap_corrected_poses--------------------" <<endl;
    unmap_submap_info_bkq = unmap_submap_info;
    pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr pos_kdtree_tmp\
        (new pcl::KdTreeFLANN<pcl::PointXYZI>());
    PointVector position_vec;
    pcl::PointXYZI posi;
    fout_dbg<<"Correct KF pose @" ;
    for (int i = 0; i < path_cor->poses.size(); i++)
    {
        geometry_msgs::PoseStamped pose_cor = path_cor->poses[i];
        int idx = int(pose_cor.header.seq);
        if (idx >= 50000) idx = -(idx - 50000);
        auto iter = unmap_submap_info.find(idx);
        if (iter != unmap_submap_info.end())
        {
            auto &submap_info = iter->second;
            M3D R1 = submap_info.lidar_pose_rotM;
            V3D t1 = submap_info.lidar_pose_tran;
            M3D R2;   V3D t2;
            if (idx < 0 )  continue; // prior kf range
            ExtraLib::poseMsgToEigenRT(pose_cor.pose, R2, t2);
            submap_info.corr_pose_rotM = R2;
            submap_info.corr_pose_tran = t2;
            submap_info.corPoseSet = true;
            posi.x = submap_info.corr_pose_tran[0], posi.y = submap_info.corr_pose_tran[1], posi.z = submap_info.corr_pose_tran[2], posi.intensity = idx;
            fout_dbg<< " " << idx ;
            key_poses->push_back(posi);
#ifdef save_for_mapconsistency_eva
            Eigen::Matrix4d T2 = M3D_to_M4D(R1, t1);
            Eigen::Matrix4d T3 = M3D_to_M4D(R2, t2);
            for (int si = 0; si < submap_info.scan_poses_insubmap.size(); si++)
            {
                auto scan_pose = submap_info.scan_poses_insubmap[si];
                auto extrinsic = submap_info.extrinsics_insubmap[si];
                Eigen::Matrix4d T1 = M3D_to_M4D(std::get<1>(scan_pose), std::get<2>(scan_pose));
                Eigen::Matrix4d guess = T3*T2.inverse()*T1;
                Eigen::Quaterniond current_q(guess.block<3,3>(0,0));
                Eigen::Vector3d t_scan(guess(0,3), guess(1,3), guess(2,3));

                Eigen::Quaterniond q_off(std::get<1>(extrinsic));
                Eigen::Vector3d t_off = std::get<2>(extrinsic);

                scan_pose_cor_file << fixed << setprecision(6) << std::to_string(std::get<0>(scan_pose)) << " "
                                   << setprecision(7) << t_scan(0) << " "
                                   << t_scan(1) << " " << t_scan(2)
                                   << " " << current_q.x() << " " << current_q.y() << " "
                                   << current_q.z() << " " << current_q.w() << " "
                                   << t_off(0) << " " << t_off(1) << " " << t_off(2)
                                   << " " << q_off.x() << " " << q_off.y() << " "
                                   << q_off.z() << " " << q_off.w() << std::endl;
            }
#endif
        }
    }
    fout_dbg<< "----------------------------------------" << endl ;
    pos_kdtree_tmp->setInputCloud(key_poses);
    pos_kdtree = pos_kdtree_tmp->makeShared();
}
void recover_unmap_submap_info()
{
    unmap_submap_info = unmap_submap_info_bkq;
}

int last_occur_id = 0;
std::unordered_map<int, bool> submap_onikdtree_flag;
void update_ikdtree_with_submap_corrected()
{
    if (!first_correction_set) return;

    if (submap_id > 1 && submap_id - last_occur_id > 0)
    {
        last_occur_id = submap_id;
        std::vector<float> pointSearchSqDis;
        std::vector<int> pointIndices;

        pcl::PointXYZI pos_lc;
        pos_lc.x = kf.get_x().pos(0);
        pos_lc.y = kf.get_x().pos(1);
        pos_lc.z = kf.get_x().pos(2);

        bool use_prior = true;
if (multisession_mode == 1)
{
        pos_kdtree_prior->radiusSearch(pos_lc, dy_mapretrival_range, pointIndices, pointSearchSqDis);
        if (pointIndices.empty())
        {
            pos_kdtree->radiusSearch(pos_lc, dy_mapretrival_range, pointIndices, pointSearchSqDis);
            use_prior = false;
        }
        fout_dbg << "Pose @ " << pos_lc << " has " << pointIndices.size() << "near KF poses" << std::endl;
}
else
        pos_kdtree->radiusSearch(pos_lc, dy_mapretrival_range, pointIndices, pointSearchSqDis);

        PointCloudXYZI::Ptr correctd_cloud_submap_local(new PointCloudXYZI());
        double ptcor_start_time = omp_get_wtime();
        fout_dbg<<"Accumulate "<<pointIndices.size() << " keyframes" <<endl;
        for (auto &aidx : pointIndices)
        {
            int kf_index;
if (multisession_mode == 1)
            kf_index = use_prior?key_poses_prior->points[aidx].intensity:key_poses->points[aidx].intensity;
else
            kf_index = key_poses->points[aidx].intensity;

//      fout_dbg<< " (" << kf_index <<",";
            if (submap_onikdtree_flag[kf_index] == true)
                continue;

            auto iter = unmap_submap_info.find(kf_index);
            if (iter == unmap_submap_info.end())
                continue;
            auto &submap_info = iter->second;

            if (!submap_info.corPoseSet || !submap_info.oriPoseSet)
                continue;

            int cloud_size = submap_info.cloud_ontree.size();
//      fout_dbg << cloud_size << ") = " << RotMtoEuler(submap_info.lidar_pose_rotM).transpose() << " " << RotMtoEuler(submap_info.corr_pose_rotM).transpose() << " ";
            submap_onikdtree_flag[kf_index] = true;

            PointCloudXYZI::Ptr laserCloudWorldCorrected(new PointCloudXYZI(cloud_size, 1));

            if (kf_index < 0 )
            {
                for (int i = 0; i < cloud_size; i++)
                    laserCloudWorldCorrected->push_back(submap_info.cloud_ontree[i]);
            }
            else
            {
                for (int i = 0; i < cloud_size; i++)
                    correctLidarPoints(&submap_info.cloud_ontree[i], \
                                   &laserCloudWorldCorrected->points[i], \
                                   submap_info.lidar_pose_rotM, submap_info.lidar_pose_tran,\
                                   submap_info.corr_pose_rotM , submap_info.corr_pose_tran);
            }
            *correctd_cloud_submap_local +=  *laserCloudWorldCorrected;
        }
        PointCloudXYZI::Ptr cloud_ds(new PointCloudXYZI());
        pcl::VoxelGrid<PointType> sor;
        float leafsize = 0.75;
        sor.setLeafSize(leafsize, leafsize, leafsize);
        sor.setInputCloud(correctd_cloud_submap_local);
        sor.filter(*cloud_ds);

        fout_dbg<<"Accumulate "<<correctd_cloud_submap_local->size()<<" pts"<< ", aft downsample "  << cloud_ds->size() <<" pts"<< std::endl;
        fout_dbg<<"Correct points takes "<<omp_get_wtime()-ptcor_start_time<<" s"<<endl;
        PointToAddHistorical.push_back(cloud_ds);

        if (pubLaserCloudFullCor.getNumSubscribers() != 0)
        {
            *correctd_cloud_submap += *cloud_ds;
            sensor_msgs::PointCloud2 laserCloudmsg;
            pcl::toROSMsg(*correctd_cloud_submap, laserCloudmsg);
            laserCloudmsg.header.stamp = ros::Time().fromSec(last_timestamp_lidar);
            laserCloudmsg.header.frame_id = "camera_init";
            pubLaserCloudFullCor.publish(laserCloudmsg);
        }
    }
}

V3D last_update_pose(0,0,0);
int correctpt_counter = 0, ikdtreebuild_counter = 0;
float correctpt_timeavg = 0.0f, ikdtreebuild_timeavg = 0.0f;
int correctd_submapsize = 0;
int init_accu_pt_containner_size = 500000;
PointCloudXYZI::Ptr correctd_cloud_rebuild(new PointCloudXYZI(init_accu_pt_containner_size, 1));
void ikdtree_rebuild()
{
#ifdef save_for_mapconsistency_eva
    std::fstream scan_pose_cor_file;
#endif
    ros::Rate rate(20);
    while (ros::ok() )
    {
        rate.sleep();
        sig_ikdtreeptr.notify_all();  // allow fastlio thread do its things

        if (last_submap_id == submap_id) continue; //no new kf, do nothing

        auto start = std::chrono::system_clock::now();
        update_submap_info();
        update_ikdtree_with_submap_corrected();
        if ( path_buffer.empty() || unmap_submap_info.empty())
        {
            auto end = std::chrono::system_clock::now();
            std::chrono::duration<double, std::milli> elapsed_ms = std::chrono::duration<double,std::milli>(end - start);
            time_ikdrebuild_thread << elapsed_ms.count() << std::endl;
            continue;
        }
        V3D curr_update_pose(path_buffer.front()->poses.back().pose.position.x,\
                             path_buffer.front()->poses.back().pose.position.y,\
                             path_buffer.front()->poses.back().pose.position.z);

#ifdef save_for_mapconsistency_eva
        scan_pose_cor_file.open(scanpose_cor_filename, ios::out | ios::trunc);
        set_submap_corrected_poses(path_buffer.front(), scan_pose_cor_file);
        scan_pose_cor_file.close();
        path_buffer.pop_front();
        //continue;  // disable LTA - pose correction and ikdtree rebuild
        float update_dis = (curr_update_pose-last_update_pose).norm();
        if (update_dis < correction_dis_interval && first_correction_set)
        {
            auto end = std::chrono::system_clock::now();
            std::chrono::duration<double, std::milli> elapsed_ms = std::chrono::duration<double,std::milli>(end - start);
            time_ikdrebuild_thread << elapsed_ms.count() << std::endl;
            continue;
        }
#else
        float update_dis = (curr_update_pose-last_update_pose).norm();
        if (update_dis < correction_dis_interval && first_correction_set)
        {
            path_buffer.pop_front();
            auto end = std::chrono::system_clock::now();
            std::chrono::duration<double, std::milli> elapsed_ms = std::chrono::duration<double,std::milli>(end - start);
            time_ikdrebuild_thread << elapsed_ms.count() << std::endl;
            continue;
        }
        set_submap_corrected_poses(path_buffer.front());
        path_buffer.pop_front();
#endif

        int submap_size = unmap_submap_info.size();//submap_info_buffer.size();
        std::vector<float> pointSearchSqDis;
        std::vector<int> pointIndices;

        pcl::PointXYZI pos_lc;
        int pos_id_lc = submap_size-1;
        while(!unmap_submap_info[pos_id_lc].corPoseSet)
            pos_id_lc--;
        //assert(unmap_submap_info[pos_id_correct].corPoseSet == true);
        pos_lc.x = unmap_submap_info[pos_id_lc].corr_pose_tran[0];
        pos_lc.y = unmap_submap_info[pos_id_lc].corr_pose_tran[1];
        pos_lc.z = unmap_submap_info[pos_id_lc].corr_pose_tran[2];

        bool use_prior = true;
if (multisession_mode == 1)
{
        pos_kdtree_prior->radiusSearch(pos_lc, dy_mapretrival_range, pointIndices, pointSearchSqDis);
        if (pointIndices.empty())
        {
            pos_kdtree->radiusSearch(pos_lc, dy_mapretrival_range, pointIndices, pointSearchSqDis);
            use_prior = false;
        }
}
else
        pos_kdtree->radiusSearch(pos_lc, dy_mapretrival_range, pointIndices, pointSearchSqDis);

        fout_dbg<<"Surrounding (@ " << unmap_submap_info[pos_id_lc].corr_pose_tran.transpose() << ") KF pose number: "<< pointIndices.size() << " including: ";
        std::vector<int> submap_ids_vec;
        double ptcor_start_time = omp_get_wtime();
        int accu_pt_counter = 0;

        int kf_index;
        for (auto &pind : pointIndices)
        {
if (multisession_mode == 1)
            kf_index = use_prior?key_poses_prior->points[pind].intensity:key_poses->points[pind].intensity;
else
            kf_index = key_poses->points[pind].intensity;

            fout_dbg<< " " << kf_index;
            auto iter = unmap_submap_info.find(kf_index);
            if (iter == unmap_submap_info.end())
                continue;

            auto &submap_info = iter->second;

            if (!submap_info.corPoseSet || !submap_info.oriPoseSet)
                continue;

            int cloud_size = submap_info.cloud_ontree.size();
            if (cloud_size==0) continue;
            submap_ids_vec.push_back(kf_index);
//            PointCloudXYZI::Ptr laserCloudWorldCorrected(new PointCloudXYZI(cloud_size, 1));
//            laserCloudWorldCorrected->points.reserve(cloud_size);

            if (accu_pt_counter + cloud_size > init_accu_pt_containner_size)
                continue;

            if (kf_index < 0 )
            {
                for (int i = 0; i < cloud_size; i++) // Prior map no need to correct
                    correctd_cloud_rebuild->points[accu_pt_counter+i] = submap_info.cloud_ontree[i];
                accu_pt_counter += cloud_size;
            }
            else
            {
                for (int i = 0; i < cloud_size; i++)
                    correctLidarPoints(&submap_info.cloud_ontree[i], \
                                       &correctd_cloud_rebuild->points[accu_pt_counter+i], \
                                       submap_info.lidar_pose_rotM, submap_info.lidar_pose_tran,\
                                       submap_info.corr_pose_rotM , submap_info.corr_pose_tran);
                accu_pt_counter += cloud_size;
            }
//            *correctd_cloud_rebuild +=  *laserCloudWorldCorrected;
        }
        correctd_cloud_rebuild->resize(accu_pt_counter);
        fout_dbg << "Accumulated points number: " << correctd_cloud_rebuild->size() << endl;
        correctpt_counter++;
        fout_dbg<<"Correct points takes :"<<omp_get_wtime()-ptcor_start_time<<"s"<<endl;
        correctpt_timeavg += omp_get_wtime()-ptcor_start_time;
        fout_dbg<<"[Time] Correcting pts cost avg :"<<correctpt_timeavg/float(correctpt_counter) << "s" <<endl;

        if (accu_pt_counter == 0)
        {
            auto end = std::chrono::system_clock::now();
            std::chrono::duration<double, std::milli> elapsed_ms = std::chrono::duration<double,std::milli>(end - start);
            time_ikdrebuild_thread << elapsed_ms.count() << std::endl;
            continue;
        }
        if (correctd_cloud_rebuild->size() > 150000)
        {
            int num_pt_cur = correctd_cloud_rebuild->size();
            std::vector<int> indices;
            int sample_gap = ceil(double(num_pt_cur)/double(150000));
            for (int i = 0; i<correctd_cloud_rebuild->size(); i+=sample_gap)
                indices.push_back(i);

            PointCloudXYZI::Ptr correctd_cloud_submap_tmp(new PointCloudXYZI());
            pcl::copyPointCloud(*correctd_cloud_rebuild,indices,*correctd_cloud_submap_tmp);
            correctd_cloud_rebuild = correctd_cloud_submap_tmp;
            fout_dbg << "After downsampling, size: " << correctd_cloud_rebuild->size() << endl;
        }

        correctd_submapsize += correctd_cloud_rebuild->size();

        if(ikdtree_swapptr->Root_Node == nullptr)
        {
            ikdtree_swapptr->set_downsample_param(filter_size_map_min);
            double build_start_time = omp_get_wtime();
            ikdtree_swapptr->Build(correctd_cloud_rebuild->points);
            fout_dbg<<"Ikdtree rebuild takes :"<<omp_get_wtime()-build_start_time<<"s"<<endl;
            ikdtreebuild_timeavg += omp_get_wtime()-build_start_time;
            ikdtreebuild_counter++;
            fout_dbg<<"[Time] Ikdtree rebuild time cost avg :"<<ikdtreebuild_timeavg/float(ikdtreebuild_counter)<<"s"<<endl;
        }

        M3D Ricp = M3D::Identity();
        V3D ticp(0,0,0);

        M3D R1;
        V3D t1;
        std_msgs::Float32MultiArrayPtr notification_msg (new std_msgs::Float32MultiArray());
        notification_msg->data.push_back(1);
        pubOdomCorrection.publish(notification_msg);
        std_msgs::Float64MultiArrayPtr notification_msg2 (new std_msgs::Float64MultiArray());
        notification_msg2->data.push_back(1);
        pubOdomCorrection2.publish(notification_msg2);

        bool correction_succeed = true;
        {
            holding_for_ikdtreerebuild = true;
            unique_lock<std::mutex> my_unique_lock(mtx_ikdtreeptr);

            M3D R2, R3;
            V3D t2, t3;
            fout_dbg<<"KF pose id at LC: "<< pos_id_lc << ", current submap_id: " << submap_id << endl;
            if (unmap_submap_info[pos_id_lc].oriPoseSet && unmap_submap_info[pos_id_lc].corPoseSet)
            {
                R2 = unmap_submap_info[pos_id_lc].lidar_pose_rotM;
                t2 = unmap_submap_info[pos_id_lc].lidar_pose_tran;
                R3 = unmap_submap_info[pos_id_lc].corr_pose_rotM;
                t3 = unmap_submap_info[pos_id_lc].corr_pose_tran;
            }

            R1 = tmp_state.rot.toRotationMatrix();
            t1 = tmp_state.pos;
            M3D Roff = tmp_state.offset_R_L_I.toRotationMatrix() ;
            V3D toff = tmp_state.offset_T_L_I;
            MD(4,4) Toff = M3D_to_M4D(Roff, toff);
            MD(4,4) T1 = M3D_to_M4D(R1, t1);
            MD(4,4) T2 = M3D_to_M4D(R2, t2);
            MD(4,4) T3 = M3D_to_M4D(R3, t3);
            MD(4,4) guess = T3*T2.inverse()*T1*Toff;

            M3D Rguess = guess.block<3,3>(0,0);
            V3D tguess = guess.block<3,1>(0,3);

            if (!refine_with_pt2pt_icp(feats_down_body, correction_ver_thr, 10, ikdtree_swapptr, Ricp, ticp,\
                                       Rguess, tguess, pos_id_lc, fout_dbg))
            {
                sig_ikdtreeptr.notify_all();
                holding_for_ikdtreerebuild = false;
                notification_msg->data.clear(); notification_msg->data.push_back(0);
                pubOdomCorrection.publish(notification_msg);
                notification_msg2->data.clear(); notification_msg2->data.push_back(0);
                pubOdomCorrection2.publish(notification_msg2);

                correctd_cloud_rebuild->clear();
                correctd_cloud_rebuild->resize(init_accu_pt_containner_size);
                ikdtree_swapptr = boost::make_shared<KD_TREE<PointType>>();
                recover_unmap_submap_info();
                correction_succeed = false;
                fout_dbg<<"Ricp :"<< RotMtoEuler(Ricp).transpose() << endl;
                fout_dbg<<"ticp :"<< ticp.transpose() << endl;
                fout_dbg<<"------------------bad correction, skip!-----------------"<<endl;
            }
            else
            {
                MD(4,4) Ticp = M3D_to_M4D(Ricp, ticp);
                notification_msg->data.clear();
                notification_msg2->data.clear();
                M3D Rvel = Ricp*R3*R2.transpose();
                set_KF_pose(kf, tmp_state, ikdtree_swapptr, feats_down_body, guess, Ticp, guess*Toff.inverse(), \
                           Rvel, pos_id_lc, notification_msg, notification_msg2, fout_dbg);
                fout_dbg << "After set_KF_pose, KF pose: " << kf.get_x().pos.transpose() << endl;
                std::swap(ikdtree_ptr, ikdtree_swapptr);
                fout_dbg << "ikdtree_ptr -> ikdtree_swapptr" << endl;
                holding_for_ikdtreerebuild = false;
                sig_ikdtreeptr.notify_all();
            }
        }
        if (!correction_succeed)
        {
            auto end = std::chrono::system_clock::now();
            std::chrono::duration<double, std::milli> elapsed_ms = std::chrono::duration<double,std::milli>(end - start);
            time_ikdrebuild_thread << elapsed_ms.count() << std::endl;
            continue;
        }

        fout_dbg<<"------------------one correction done-----------------"<<endl;
        pubOdomCorrection.publish(notification_msg);
        pubOdomCorrection2.publish(notification_msg2);
        std_msgs::UInt64 jump_time_msg;
        jump_time_msg.data = curr_Mea_lidarbegtime_NS;
        pubTimeCorrection.publish(jump_time_msg);

        if (pubLaserCloudFullCor.getNumSubscribers() != 0)
        {
            correctd_cloud_submap->clear();
            *correctd_cloud_submap += *correctd_cloud_rebuild;
            sensor_msgs::PointCloud2 laserCloudmsg;
            pcl::toROSMsg(*correctd_cloud_submap, laserCloudmsg);
            laserCloudmsg.header.stamp = ros::Time().fromSec(last_timestamp_lidar);//ros::Time::now();
            laserCloudmsg.header.frame_id = "camera_init";
            pubLaserCloudFullCor.publish(laserCloudmsg);
        }

        ExtraLib::pubCorrectionIds(pubCorrectionId, tmp_state.pos, submap_id);
        ikdtree_swapptr = boost::make_shared<KD_TREE<PointType>>();

        submap_onikdtree_flag.clear(); //reset flag when new pose correction done
        for (auto aid:submap_ids_vec)
            submap_onikdtree_flag[aid] = true;

        last_update_pose = curr_update_pose;
        first_correction_set = true;
        fout_dbg << "First correction set." << endl;
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double, std::milli> elapsed_ms1 = std::chrono::duration<double,std::milli>(end - start);
        time_ikdrebuild_thread << elapsed_ms1.count() << std::endl;

        correctd_cloud_rebuild->clear();
        correctd_cloud_rebuild->resize(init_accu_pt_containner_size);
    }
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
void publish_frame_world(const ros::Publisher & pubLaserCloudFullRes, const double curr_timestamp_lidar)
{
    PointCloudXYZI::Ptr laserCloudFullRes(dense_map_en ? feats_undistort : feats_down_body);
    int size = laserCloudFullRes->points.size();
    PointCloudXYZI::Ptr laserCloudWorld( \
                    new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
                            &laserCloudWorld->points[i]);
    }

    *pcl_wait_pub += *laserCloudWorld;

    if(1) // (publish_count >= PUBFRAME_PERIOD)
    {
#ifdef as_node
        sensor_msgs::PointCloud2Ptr laserCloudmsg (new sensor_msgs::PointCloud2());
        pcl::toROSMsg(*pcl_wait_pub, *laserCloudmsg);
        laserCloudmsg->header.stamp = ros::Time().fromSec(curr_timestamp_lidar);//ros::Time::now();
        laserCloudmsg->header.frame_id = "camera_init";
#else
        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*pcl_wait_pub, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(last_timestamp_lidar);//ros::Time::now();
        laserCloudmsg.header.frame_id = "camera_init";
#endif
        pubLaserCloudFullRes.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
        pcl_wait_pub->clear();
    }
}

void publish_effect_world(const ros::Publisher & pubLaserCloudEffect)
{
    PointCloudXYZI::Ptr laserCloudWorld( \
                    new PointCloudXYZI(effct_feat_num, 1));
    for (int i = 0; i < effct_feat_num; i++)
    {
        RGBpointBodyToWorld(&laserCloudOri->points[i], \
                            &laserCloudWorld->points[i]);
    }
    sensor_msgs::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = ros::Time::now();//.fromSec(last_timestamp_lidar);
    laserCloudFullRes3.header.frame_id = "camera_init";
    pubLaserCloudEffect.publish(laserCloudFullRes3);
}

void publish_map(const ros::Publisher & pubLaserCloudMap)
{
    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*featsFromMap, laserCloudMap);
    laserCloudMap.header.stamp = ros::Time::now();
    laserCloudMap.header.frame_id = "camera_init";
    pubLaserCloudMap.publish(laserCloudMap);
}

void publish_odometry(const ros::Publisher & pubOdomAftMapped, const double curr_timestamp_lidar)
{
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "aft_mapped";
    odomAftMapped.header.stamp = /*ros::Time::now();*/ros::Time().fromSec(curr_timestamp_lidar);
    set_posestamp(odomAftMapped.pose.pose);

    // Publish TF
    static tf::TransformBroadcaster br;
    tf::Quaternion q_tmp(odomAftMapped.pose.pose.orientation.x, odomAftMapped.pose.pose.orientation.y, odomAftMapped.pose.pose.orientation.z,\
                         odomAftMapped.pose.pose.orientation.w);
    tf::Transform t_odom_to_maporigin = tf::Transform(q_tmp,
        tf::Vector3(odomAftMapped.pose.pose.position.x, odomAftMapped.pose.pose.position.y, odomAftMapped.pose.pose.position.z));
    tf::StampedTransform trans_odom_to_maporigin = tf::StampedTransform(t_odom_to_maporigin, odomAftMapped.header.stamp, "camera_init", "odom");
    br.sendTransform(trans_odom_to_maporigin);
    pubOdomAftMapped.publish(odomAftMapped);
}

void publish_mavros(const ros::Publisher & mavros_pose_publisher)
{
    msg_body_pose.header.stamp = ros::Time::now();
    msg_body_pose.header.frame_id = "camera_odom_frame";
    set_posestamp(msg_body_pose.pose);
    mavros_pose_publisher.publish(msg_body_pose);
}

void publish_path(const ros::Publisher pubPath)
{
    set_posestamp(msg_body_pose.pose);
    msg_body_pose.header.stamp = ros::Time::now();
    msg_body_pose.header.frame_id = "camera_init";
    path.poses.push_back(msg_body_pose);
    pubPath.publish(path);
}

void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    double match_start = omp_get_wtime();
    laserCloudOri->clear();
    corr_normvect->clear();
    total_residual = 0.0;

    /** closest surface search and residual computation **/
    #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
        #pragma omp parallel for
    #endif
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body  = feats_down_body->points[i];
        PointType &point_world = feats_down_world->points[i];
        //double search_start = omp_get_wtime();
        /* transform to world frame */
        //pointBodyToWorld_ikfom(&point_body, &point_world, s);
        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);
        auto &points_near = Nearest_Points[i];

        if (ekfom_data.converge)
        {
            /** Find the closest surfaces in the map **/
            ikdtree_ptr->Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);

            if (points_near.size() < NUM_MATCH_POINTS)
            {
                point_selected_surf[i] = false;
            }
            else
            {
                point_selected_surf[i] = pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
            }
        }
//        fout_dbg << " ( " << i << "," << point_body.x << " " << point_body.y << "|" << point_world.x << " " << point_world.y << " ) " << std::endl;

        //kdtree_search_time += omp_get_wtime() - search_start;

        if (!point_selected_surf[i]) continue;


        VF(4) pabcd;
        point_selected_surf[i] = false;
        if (esti_plane(pabcd, points_near, 0.1f)) //(planeValid)
        {

            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

            if (s > 0.9)
            {
                point_selected_surf[i] = true;
                normvec->points[i].x = pabcd(0);
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2;
                res_last[i] = abs(pd2);
            }
        }
    }

    effct_feat_num = 0;

    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i])
        {
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
            corr_normvect->points[effct_feat_num] = normvec->points[i];
            total_residual += res_last[i];
            effct_feat_num ++;
        }
    }
    res_mean_last = total_residual / effct_feat_num;
    match_time  += omp_get_wtime() - match_start;
    double solve_start_  = omp_get_wtime();
    fout_dbg << "scan to map update avg res_mean_last: " << res_mean_last << " KF pose: " << s.pos.transpose() << std::endl;
    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    //MatrixXd H(effct_feat_num, 23);
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12); //23
    ekfom_data.h.resize(effct_feat_num); // = VectorXd::Zero(effct_feat_num);
    //VectorXd meas_vec(effct_feat_num);

    for (int i = 0; i < effct_feat_num; i++)
    {
        const PointType &laser_p  = laserCloudOri->points[i];
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat<<SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.conjugate() *norm_vec);
        V3D A(point_crossmat * C); // s.rot.conjugate() * norm_vec);
        V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); //s.rot.conjugate()*norm_vec);
        //H.row(i) = Eigen::Matrix<double, 1, 23>::Zero();
        ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        //ekfom_data.h_x.block<1, 3>(i, 6) << VEC_FROM_ARRAY(A);
        //ekfom_data.h_x.block<1, 6>(i, 17) << VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);

        /*** Measuremnt: distance to the closest surface/corner ***/
        //meas_vec(i) = - norm_p.intensity;
        ekfom_data.h(i) = -norm_p.intensity;
    }
    //ekfom_data.h_x =H;
    solve_time += omp_get_wtime() - solve_start_;
    //return meas_vec;
}

PointCloudXYZI::Ptr lc_on_prior(new PointCloudXYZI());
void loopclosure_onprior_cbk(const std_msgs::Int32ConstPtr &id_msg)
{
    int lc_idx_onprior = -(id_msg->data+1); // use minor sign for prior
    lc_on_prior->points = unmap_submap_info[lc_idx_onprior].cloud_ontree;
}

void pub_prior_map_one_time(const ros::Publisher & a_publisher,  PointCloudXYZI::Ptr &map)
{
    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*map, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time::now();
    laserCloudmsg.header.frame_id = "camera_init";
    a_publisher.publish(laserCloudmsg);
    map->clear();
}

void load_prior_map_and_info(PointCloudXYZI::Ptr &priormap)
{
    std::cout << "[LoadPrior]: loading prior map. " << std::endl;
    double tload = omp_get_wtime();
    PointCloudXYZI::Ptr datacloud(new PointCloudXYZI());
    std::string pcd_name = save_directory + "map_prior/clouds_corrected.pcd";
    pcl::io::loadPCDFile(pcd_name, *datacloud);

    PointCloudXYZI::Ptr tmpmap(new PointCloudXYZI());
    for(int j = 0; j < datacloud->points.size(); j++)
    {
        auto a_pt = datacloud->points[j];
        if(fabs(a_pt.x - (-1010.1)) < 0.01)  // separation flag
        {
            SubmapInfo tmp_submapinfo;
            int tmp_id = -(a_pt.y+1);
            int cloud_size = a_pt.z;
            if (cloud_size != 0)
            {
                tmpmap->clear();
                for (int k = 0; k < cloud_size; k++)
                {
                    j++;
                    tmpmap->push_back(datacloud->points[j]);
                }
            }

            j++;
            a_pt = datacloud->points[j];
            tmp_submapinfo.corr_pose_rotM = ExtraLib::eulToRotM(a_pt.x, a_pt.y, a_pt.z) ;
            j++;
            a_pt = datacloud->points[j];
            tmp_submapinfo.corr_pose_tran(0) = a_pt.x; tmp_submapinfo.corr_pose_tran(1) = a_pt.y; tmp_submapinfo.corr_pose_tran(2) = a_pt.z;
            pcl::PointXYZI a_posi; a_posi.x = a_pt.x; a_posi.y = a_pt.y; a_posi.z = a_pt.z; a_posi.intensity = tmp_id;
            key_poses_prior->push_back(a_posi);
            tmp_submapinfo.corPoseSet = true;

            j++;
            a_pt = datacloud->points[j];
            tmp_submapinfo.lidar_pose_rotM = ExtraLib::eulToRotM(a_pt.x, a_pt.y, a_pt.z);
            j++;
            a_pt = datacloud->points[j];
            tmp_submapinfo.lidar_pose_tran(0) = a_pt.x; tmp_submapinfo.lidar_pose_tran(1) = a_pt.y; tmp_submapinfo.lidar_pose_tran(2) = a_pt.z;
            tmp_submapinfo.oriPoseSet = true;

            if (cloud_size != 0)
            {
                *priormap += *tmpmap;
                tmp_submapinfo.cloud_ontree = tmpmap->points;
                unmap_submap_info[tmp_id] = tmp_submapinfo;
            }

            PointCloudXYZI::Ptr empty_cloud (new PointCloudXYZI());
            tmpmap = empty_cloud;
            continue;
        }
        tmpmap->push_back(a_pt);
    }
    std::cout << "[LoadPrior]: loading prior map done, using "  << 1000*(omp_get_wtime() - tload) << " ms" << std::endl;
    std::cout << "[LoadPrior]: prior submap number "  << key_poses_prior->size() << std::endl;
    std::cout << "[LoadPrior]: total prior map size "  << priormap->size() << std::endl;
    pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr pos_kdtree_tmp(new pcl::KdTreeFLANN<pcl::PointXYZI>());
    pos_kdtree_tmp->setInputCloud(key_poses_prior);
    pos_kdtree_prior = pos_kdtree_tmp->makeShared();
}


#ifdef as_node
int mainLIOFunction()
{
    int argc; char** argv;
#else
int main(int argc, char** argv)
{
#endif
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;

    nh.param<bool>("dense_map_enable",dense_map_en,1);
    nh.param<int>("max_iteration",NUM_MAX_ITERATIONS,4);
    nh.param<string>("map_file_path",map_file_path,"");
    nh.param<string>("common/lid_topic",lid_topic,"/livox/lidar");
    nh.param<string>("common/imu_topic", imu_topic,"/livox/imu");
    nh.param<double>("common/time_offset_from_lidar", time_offset_from_lidar,0.0f);
    nh.param<double>("filter_size_corner",filter_size_corner_min,0.5);
    nh.param<double>("filter_size_surf",filter_size_surf_min,0.5);
    nh.param<double>("filter_size_map",filter_size_map_min,0.5);
    nh.param<double>("cube_side_length",cube_len,200);
    nh.param<float>("mapping/det_range",DET_RANGE,300.f);
    nh.param<double>("mapping/fov_degree",fov_deg,180);
    nh.param<double>("mapping/gyr_cov",gyr_cov,0.1);
    nh.param<double>("mapping/acc_cov",acc_cov,0.1);
    nh.param<double>("mapping/b_gyr_cov",b_gyr_cov,0.0001);
    nh.param<double>("mapping/b_acc_cov",b_acc_cov,0.0001);
    nh.param<double>("preprocess/blind", p_pre->blind, 0.01);
    nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA);
    nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);
    nh.param<int>("point_filter_num", p_pre->point_filter_num, 2);
    nh.param<bool>("feature_extract_enable", p_pre->feature_enabled, 0);
    nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>());
    nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>());
    nh.param<int>("multisession_mode", multisession_mode, 0);
    nh.param<double>("correction_ver_thr", correction_ver_thr, 0.45);
    nh.param<string>("SaveDir", save_directory, "");
    nh.param<double>("correction_dis_interval", correction_dis_interval, 50);
#ifdef save_for_mapconsistency_eva
    p_pre->point_filter_num = 1;
    scanpose_cor_filename = save_directory + "scanposes_corrected.txt";
#endif
    cout<<"p_pre->lidar_type "<<p_pre->lidar_type<<endl;

    path.header.stamp    = ros::Time::now();
    path.header.frame_id ="camera_init";

    /*** variables definition ***/
    //PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));
    int effect_feat_num = 0, frame_num = 0;
    double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
    bool flg_EKF_converged, EKF_stop_flg = 0;

    FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
    HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

    _featsArray.reset(new PointCloudXYZI());

    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);

    shared_ptr<ImuProcess> p_imu(new ImuProcess());
    // p_imu->set_extrinsic(V3D(0.04165, 0.02326, -0.0284));   //avia
    // p_imu->set_extrinsic(V3D(0.05512, 0.02226, -0.0297));   //horizon
    Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);
    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
    p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
    p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));
    if (p_pre->lidar_type == XGRIDS) p_imu->disable_undistort = true;

    double epsi[23] = {0.001};
    fill(epsi, epsi+23, 0.001);
    kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);
    /*** debug record ***/
    FILE *fp;
    string pos_log_dir = root_dir + "/Log/pos_log.txt";
    fp = fopen(pos_log_dir.c_str(),"w");

    fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"),ios::out);
    fout_out.open(DEBUG_FILE_DIR("mat_out.txt"),ios::out);
    fout_dbg.open(save_directory + "lio_debug.txt",ios::out);
    if (fout_pre && fout_out)
        cout << "~~~~"<<ROOT_DIR<<" file opened" << endl;
    else
        cout << "~~~~"<<ROOT_DIR<<" doesn't exist" << endl;

    time_ikdrebuild_thread.open(save_directory + "times_ikdrebuild_LTAOM.txt",ios::out);
    time_ikdrebuild_thread.precision(std::numeric_limits<double>::max_digits10);

#ifdef save_for_mapconsistency_eva
    rosbag::Bag bag;
    bag.open(save_directory + "undistoted_scans.bag", rosbag::bagmode::Write);
#endif

#ifdef LoadBag
    std::thread loadbag {msg_callbacks};
#else

    /*** ROS subscribe initialization ***/
    ros::Subscriber sub_pcl = p_pre->lidar_type == AVIA ? \
        nh.subscribe(lid_topic, 200000, livox_pcl_cbk) : \
        nh.subscribe(lid_topic, 200000, standard_pcl_cbk);
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);
#endif
    ros::Subscriber sub_submap_id = nh.subscribe<std_msgs::Int32>("/submap_ids", 100, submap_id_cbk);
    ros::Subscriber sub_submap_pose = nh.subscribe<nav_msgs::Odometry>("/submap_pose", 100, submap_pose_cbk);
#ifndef offline
    ros::Subscriber sub_path_corrected = nh.subscribe<nav_msgs::Path>("/aft_pgo_path", 100, path_cor_cbk);
#endif

    pubLaserCloudFullCor = nh.advertise<sensor_msgs::PointCloud2>("/cloud_corrected", 100);
//    pubOdomLargeJump = nh.advertise<nav_msgs::Odometry>("/odom_largejump", 10);
    pubOdomCorrection = nh.advertise<std_msgs::Float32MultiArray>("/odom_correction_info", 10);
    pubOdomCorrection2 = nh.advertise<std_msgs::Float64MultiArray>("/odom_correction_info64", 10);
    pubTimeCorrection = nh.advertise<std_msgs::UInt64>("/time_correction", 10);
    pubCorrectionId = nh.advertise<visualization_msgs::MarkerArray>("/ids_corr", 10);
//if (multisession_mode == 1){
    ros::Publisher pubPriorMap = nh.advertise<sensor_msgs::PointCloud2>("/cloud_prior", 100);
    ros::Publisher pubLCOnPriorMap = nh.advertise<sensor_msgs::PointCloud2>("/lc_cloud_prior", 100);
    ros::Subscriber sub_lc_onprior_id = nh.subscribe<std_msgs::Int32>("/ids_lc_onprior", 100, loopclosure_onprior_cbk);
//}
    ros::Publisher pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>
        ("/cloud_registered", 10000);
    ros::Publisher pubLaserCloudEffect  = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_effected", 100);
    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>
            ("/Laser_map", 100);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>
        ("/aft_mapped_to_init", 10000);
    ros::Publisher pubPath          = nh.advertise<nav_msgs::Path>
            ("/path", 10);
    std::thread extra_thread_for_rebuild {ikdtree_rebuild};

#ifdef DEPLOY
    ros::Publisher mavros_pose_publisher = nh.advertise<geometry_msgs::PoseStamped>("/mavros/vision_pose/pose", 10);
#endif
    PointCloudXYZI::Ptr priormap(new PointCloudXYZI());
if (multisession_mode == 1)
{
    load_prior_map_and_info(priormap);
}
//------------------------------------------------------------------------------------------------------
    signal(SIGINT, SigHandle);
    ros::Rate rate(5000);
    bool status = ros::ok();
    while (status)
    {
        if (flg_exit) break;
        ros::spinOnce();
if (multisession_mode == 1)
{
        if (!priormap->empty()) pub_prior_map_one_time(pubPriorMap, priormap);
}

        if(sync_packages(Measures))
        {
            if (flg_reset)
            {
                ROS_WARN("reset when rosbag play back");
                p_imu->Reset();
                flg_reset = false;
                continue;
            }
            double t0,t1,t2,t3,t4,t5,match_start, solve_start, svd_time;

            match_time = 0;
            kdtree_search_time = 0.0;
            solve_time = 0;
            solve_const_H_time = 0;
            svd_time   = 0;
            fout_dbg << "--------------------------------------------------------frame_num " << frame_num << std::endl;
            t0 = omp_get_wtime();
            {
                unique_lock<std::mutex> my_unique_lock(mtx_ikdtreeptr);
                sig_ikdtreeptr.wait(my_unique_lock, []{return !holding_for_ikdtreerebuild;});
                p_imu->Process(Measures, kf, feats_undistort);
            }
            state_point = kf.get_x();
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                first_lidar_time = Measures.lidar_beg_time;
                p_imu->first_lidar_time = first_lidar_time;
                cout<<"FAST-LIO not ready"<<endl;
                continue;
            }

            bool is_outrange = false;
            for (int i = 0; i < feats_undistort->size(); i++)
            {
                if(feats_undistort->points[i].x < -1000000 || feats_undistort->points[i].x > 1000000 ||
                   feats_undistort->points[i].y < -1000000 || feats_undistort->points[i].y > 1000000 ||
                   feats_undistort->points[i].z < -1000000 || feats_undistort->points[i].z > 1000000)
                {
                    cout << "[ Warn ]: point_in_body is out of reasonable range!" << endl;
                    is_outrange = true;
                    break;
                }
            }
            if (is_outrange) continue;
            ros::Time tmp_time; tmp_time.fromSec(Measures.lidar_beg_time);
            curr_Mea_lidarbegtime_NS = tmp_time.toNSec();

            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? \
                            false : true;

//        #define DEBUG_PRINT
        #ifdef DEBUG_PRINT
            //state_ikfom debug_state = kf.get_x();
            //euler_cur = RotMtoEuler(state_point.rot.toRotationMatrix());
            euler_cur = SO3ToEuler(state_point.rot);
            cout<<"current lidar time "<<Measures.lidar_beg_time<<" "<<"first lidar time "<<first_lidar_time<<endl;
            cout<<"pre-integrated states: "<<euler_cur.transpose()*57.3<<" "<<state_point.pos.transpose()<<" "<<state_point.vel.transpose()<<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<endl;
        #endif
            /*** Segment the map in lidar FOV ***/
            lasermap_fov_segment();
                // fout_out << "Before seg- tree size: " << fov_rec_before[0] << " " << fov_rec_before[1] << " " << fov_rec_before[2] << endl;
                // fout_out << "FoV seg - size : " << fov_size[0] << " " << fov_size[1] << " " << fov_size[2] << endl;
                // fout_out << "After seg - tree size: " << fov_rec_after[0] << " " << fov_rec_after[1] << " " << fov_rec_after[2] << endl;
                // cout << "Max Queue Size is : " << ikdtree.max_queue_size << endl;
                // fout_out << "Point Cache Size: " << points_cache_size << endl;
            /*** downsample the feature points in a scan ***/

            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);

            t1 = omp_get_wtime();
            feats_down_size = feats_down_body->points.size();
            /*** initialize the map kdtree ***/

            if(ikdtree_ptr->Root_Node == nullptr)
            {
                if(feats_down_size > 5)
                {
                    unique_lock<std::mutex> my_unique_lock(mtx_ikdtreeptr);
                    sig_ikdtreeptr.wait(my_unique_lock, []{return !holding_for_ikdtreerebuild;});
                    ikdtree_ptr->set_downsample_param(filter_size_map_min);
                    feats_down_world->resize(feats_down_size);
                    for(int i = 0; i < feats_down_size; i++)
                    {
                        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                    }
                    ikdtree_ptr->Build(feats_down_world->points);
                }
                continue;
            }
            int featsFromMapNum = ikdtree_ptr->validnum();
            kdtree_size_st = ikdtree_ptr->size();

            // cout<<"[ mapping ]: Raw feature num: "<<feats_undistort->points.size()<<" downsamp num "<<feats_down_size<<" Map num: "<<featsFromMapNum<<endl;

            /*** ICP and iterated Kalman filter update ***/
            normvec->resize(feats_down_size);
            feats_down_world->resize(feats_down_size);
            // VD(DIM_STATE) P_diag = state.cov.diagonal();
            // cout<<"P_pre: "<<P_diag.transpose()<<endl;

            //state_ikfom fout_state = kf.get_x();
            V3D ext_euler = SO3ToEuler(state_point.offset_R_L_I);
            fout_pre<<setw(20)<<Measures.lidar_beg_time - first_lidar_time<<" "<<euler_cur.transpose()<<" "<< state_point.pos.transpose()<<" "<<ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<< " " << state_point.vel.transpose() \
            <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<< endl;

            pointSearchInd_surf.resize(feats_down_size);
            Nearest_Points.resize(feats_down_size);
            int  rematch_num = 0;
            bool nearest_search_en = true; //

            t2 = omp_get_wtime();

            /*** iterated state estimation ***/
            #ifdef MP_EN
            // cout<<"Using multi-processor, used core number: "<<MP_PROC_NUM<<endl;
            #endif
            double t_update_start = omp_get_wtime();
            // V3D search_target_sum(0,0,0);
            // V3D search_result_sum(0,0,0);
            double solve_H_time = 0;
            {
                tmp_state = kf.get_x();
                unique_lock<std::mutex> my_unique_lock(mtx_ikdtreeptr);
                sig_ikdtreeptr.wait(my_unique_lock, []{return !holding_for_ikdtreerebuild;});
                kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
            }

            //state_ikfom updated_state = kf.get_x();
            state_point = kf.get_x();
            //euler_cur = RotMtoEuler(state_point.rot.toRotationMatrix());
            euler_cur = SO3ToEuler(state_point.rot);
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
            // cout<<"position: "<<pos_lid.transpose()<<endl;
            geoQuat.x = state_point.rot.coeffs()[0];
            geoQuat.y = state_point.rot.coeffs()[1];
            geoQuat.z = state_point.rot.coeffs()[2];
            geoQuat.w = state_point.rot.coeffs()[3];
            double t_update_end = omp_get_wtime();

            /******* Publish odometry *******/
//            ros::Time ros_time_tmp = ros::Time::fromNSec(Measures.lidar_beg_time);
            publish_odometry(pubOdomAftMapped, Measures.lidar_beg_time);

#ifdef save_for_mapconsistency_eva
            scan_poses_queue.push_back({Measures.lidar_beg_time, state_point.rot.toRotationMatrix(), state_point.pos});
            extrinsics_queue.push_back({Measures.lidar_beg_time, state_point.offset_R_L_I.toRotationMatrix(), state_point.offset_T_L_I});
            if (feats_undistort->size() > 0)
            {
                auto last_translation = state_point.pos;
                ros::Time current_time = ros::Time().fromSec(Measures.lidar_beg_time);
                sensor_msgs::PointCloud2 save_cloud;
                pcl::toROSMsg(*feats_undistort, save_cloud);
                save_cloud.header.frame_id = "camera_init";
                save_cloud.header.stamp = current_time;
                std::cout << "save frame:" << scan_poses_queue.size()
                          << ", cloud size:" << feats_undistort->size() << std::endl;
                bag.write("/cloud_undistort", current_time, save_cloud);
            }
#endif

            /*** add the feature points to map kdtree ***/
            t3 = omp_get_wtime();
            // fout_out << "Before - tree size: " << ikdtree.validnum() << endl;

            map_incremental();

            // fout_out << "After - tree size: " << ikdtree.validnum() << endl;
            t5 = omp_get_wtime();
            kdtree_size_end = ikdtree_ptr->size();

            /******* Publish points *******/
            publish_frame_world(pubLaserCloudFullRes, Measures.lidar_beg_time);
            publish_effect_world(pubLaserCloudEffect);
            // publish_map(pubLaserCloudMap);
            publish_path(pubPath);
            #ifdef DEPLOY
            publish_mavros(mavros_pose_publisher);
            #endif

            fout_dbg << "R_offset" << std::setprecision(10) << state_point.offset_R_L_I.toRotationMatrix() << std::endl;
            fout_dbg << "T_offset" << state_point.offset_T_L_I << std::endl;
            fout_dbg << "bias_a " << std::setprecision(10) << state_point.ba << std::endl;
            fout_dbg << "bias_b " << state_point.bg << std::endl;

            /*** Debug variables ***/
            frame_num ++;
            aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
            aver_time_icp = aver_time_icp * (frame_num - 1)/frame_num + (t_update_end - t_update_start) / frame_num;
            aver_time_match = aver_time_match * (frame_num - 1)/frame_num + (match_time)/frame_num;
            aver_time_incre = aver_time_incre * (frame_num - 1)/frame_num + (kdtree_incremental_time)/frame_num;
            aver_time_solve = aver_time_solve * (frame_num - 1)/frame_num + (solve_time + solve_H_time)/frame_num;
            aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1)/frame_num + solve_time / frame_num;
            // aver_time_consu = aver_time_consu * 0.9 + (t5 - t0) * 0.1;
            T1[time_log_counter] = Measures.lidar_beg_time;
            s_plot[time_log_counter] = t5 - t0;
            s_plot2[time_log_counter] = feats_down_size;
            s_plot3[time_log_counter] = kdtree_incremental_time;
            s_plot4[time_log_counter] = kdtree_search_time;
            s_plot5[time_log_counter] = kdtree_delete_counter;
            s_plot6[time_log_counter] = kdtree_delete_time;
            s_plot7[time_log_counter] = kdtree_size_st;
            s_plot8[time_log_counter] = kdtree_size_end;
            s_plot9[time_log_counter] = aver_time_consu;
            s_plot10[time_log_counter] = add_point_size;
            time_log_counter ++;
            printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f construct H: %0.6f \n",t1-t0,aver_time_match,aver_time_solve,t3-t1,t5-t3,aver_time_consu,aver_time_icp, aver_time_const_H_time);
            ext_euler = SO3ToEuler(state_point.offset_R_L_I);
            fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose()<< " " << ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<<" "<< state_point.vel.transpose() \
            <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<<" "<<feats_undistort->points.size()<<endl;
            dump_lio_state_to_log(fp);
            if (do_posecorrection)  // In case setKF not success
            {
                  do_posecorrection = false;
                  tmp_state.pos = pos_replace;
                  tmp_state.rot = rot_replace;
                  tmp_state.vel = vel_replace;
                  kf.change_x(tmp_state);
            }
        }
        status = ros::ok();
        rate.sleep();
    }
#ifndef DEPLOY
    vector<double> t, s_vec, s_vec2, s_vec3, s_vec4, s_vec5, s_vec6, s_vec7;
    FILE *fp2;
//    string log_dir = root_dir + "/Log/fast_lio_time_log.csv";
    string log_dir = save_directory + "fast_lio_time_log.csv";
    fp2 = fopen(log_dir.c_str(),"w");
    fprintf(fp2,"time_stamp, total time, scan point size, incremental time, search time, delete size, delete time, tree size st, tree size end, add point size, preprocess time\n");
    for (int i = 0;i<time_log_counter; i++)
    {
        fprintf(fp2,"%0.8f,%0.8f,%d,%0.8f,%0.8f,%d,%0.8f,%d,%d,%d,%0.8f\n",T1[i],s_plot[i],int(s_plot2[i]),s_plot3[i],s_plot4[i],int(s_plot5[i]),s_plot6[i],int(s_plot7[i]),int(s_plot8[i]), int(s_plot10[i]), s_plot11[i]);
        t.push_back(T1[i]);
        s_vec.push_back(s_plot9[i]);
        s_vec2.push_back(s_plot3[i] + s_plot6[i]);
        s_vec3.push_back(s_plot4[i]);
        s_vec5.push_back(s_plot[i]);
    }
    fclose(fp2);
#endif

    return 0;
}



bool refine_with_pt2pt_icp(const PointCloudXYZI::Ptr &feats_down_body, const float rmse_thr, const int iteration_thr, const boost::shared_ptr<KD_TREE<PointType>> &ikd_in, \
                        M3D &Ricp, V3D &ticp,  const M3D &Rguess, const V3D &tguess, const int pos_id_lc, std::ostream &fout_dbg)
{
    fout_dbg<<pos_id_lc<<"--------------------refine_with_pt2pt_icp--------------------" <<endl;

    // Align pointclouds
    double match_start_tmp = omp_get_wtime();
    int cloud_size = feats_down_body->size();
//    assert(cloud_size < 100000);
    Ricp = M3D::Identity(); ticp = V3D(0,0,0);
    MD(4,4) T_full = MD(4,4)::Identity();
    int iteration = 0;float rmse = 0.0f;
    double elapsed_ms;
    while (iteration < iteration_thr)
    {
        point_world_tmp->clear();
        point_near_tmp->clear();
        iteration++;
        int num_match = 1;
        #ifdef MP_EN
            omp_set_num_threads(MP_PROC_NUM);
            #pragma omp parallel for
        #endif
        for (int i = 0; i < cloud_size; i++)
        {
            const PointType point_body = feats_down_body->points[i];
            PointType &point_world = feats_down_world_guess->points[i];
            V3D p_body(point_body.x, point_body.y, point_body.z);
            V3D p_global(Ricp*(Rguess*p_body + tguess) + ticp);
            point_world.x = p_global(0);
            point_world.y = p_global(1);
            point_world.z = p_global(2);
            point_world.intensity = point_body.intensity;

            PointVector &points_near = PointsNearVec[i];
            vector<float> pointSearchSqDis(num_match);
            ikd_in->Nearest_Search(point_world, num_match, points_near, pointSearchSqDis);
            point_dis[i] = sqrt(pointSearchSqDis[0]);
        }

        float rmse_sum = 0.0f;
        for (int i = 0; i < cloud_size; i++)
        {
            rmse_sum += point_dis[i];
            if (point_dis[i] < rmse_thr) // Accelerate convergence
                continue;
            PointType point_near = PointsNearVec[i][0];
            PointType point_world = feats_down_world_guess->points[i];
            point_world_tmp->push_back(point_world);
            point_near_tmp->push_back(point_near);
        }
        rmse = rmse_sum/float(cloud_size);
        fout_dbg <<iteration << " iters fastlio icp residual: " << rmse << " ";
        if (rmse < rmse_thr)
        {
            elapsed_ms = 1000*(omp_get_wtime() - match_start_tmp);
            match_time_avg += elapsed_ms;
            match_count++;
            fout_dbg << elapsed_ms << "ms" << "______all done______" << " ";
            fout_dbg << "[Time] avg loop correction re-registration : " << (match_time_avg/float(match_count))/1000 << " s" << endl;
            return true;
        }

        int new_size = point_near_tmp->size();
        Eigen::Matrix<double, 3, Eigen::Dynamic> point_src (3, new_size);
        Eigen::Matrix<double, 3, Eigen::Dynamic> point_tgt (3, new_size);
        for (int i = 0; i < new_size; i++)
        {
            PointType &point_near = point_near_tmp->points[i];
            PointType &point_world = point_world_tmp->points[i];
            point_src (0, i) = point_world.x;
            point_src (1, i) = point_world.y;
            point_src (2, i) = point_world.z;
            point_tgt (0, i) = point_near.x;
            point_tgt (1, i) = point_near.y;
            point_tgt (2, i) = point_near.z;
        }
        MD(4,4) T_delta = pcl::umeyama (point_src, point_tgt, false);
        T_full = T_delta*T_full;
        Ricp = T_full.block<3,3>(0,0);
        ticp = T_full.block<3,1>(0,3);
        fout_dbg <<  iteration << " iters icp result euler: " << RotMtoEuler(Ricp).transpose();
        fout_dbg << " tran: " << ticp.transpose() << " ";
        elapsed_ms = 1000*(omp_get_wtime() - match_start_tmp);
        fout_dbg<< elapsed_ms << "ms" << "______one iteration done______ " << endl;
    }
    elapsed_ms = 1000*(omp_get_wtime() - match_start_tmp);
    fout_dbg  << elapsed_ms << "ms, " << "______reach max iterations______" <<endl;

    match_time_avg += elapsed_ms;
    match_count++;
    fout_dbg << "[Time] avg loop correction re-registration : " << (match_time_avg/float(match_count))/1000 << " s" << endl;

    if (rmse < rmse_thr)
        return true;

    return false;
}


bool refine_with_pt2_plane_icp(const PointCloudXYZI::Ptr &feats_down_body, const float rmse_thr, const int iteration_thr, const boost::shared_ptr<KD_TREE<PointType>> &ikd_in, \
                           M3D &Ricp, V3D &ticp, const M3D &Rguess, const V3D &tguess, const int pos_id_lc, std::ostream &fout_dbg){
  fout_dbg<<pos_id_lc<<"--------------------refine_with_pt2_plane_icp------" <<endl;

  M3D Rfull = M3D::Identity();
  V3D tfull(0,0,0);

  double match_start_tmp = omp_get_wtime();
  double elapsed_ms = 0.0f;
  int cloud_size = feats_down_body->size();
  pcl::PointCloud<PointType>::Ptr result_pplane(new pcl::PointCloud<PointType>(cloud_size, 1));
  int iteration = 0;
  float rmse = 0.0f;
//  fout_dbg<< "3" << std::endl;
  while (iteration < 20){
    iteration++;
//    fout_dbg<<  "iteration" << iteration << std::endl;
    int effect_num = 0;
    rmse = 0.0f;
//    pointnear_tmp->clear();
    for (int i = 0; i < cloud_size; i++)
    {
//      fout_dbg<< "i" <<i<< std::endl;
      const PointType point_src  = feats_down_body->points[i];
      PointType &point_out = result_pplane->points[i];

      Eigen::Vector3d p_body(point_src.x, point_src.y, point_src.z);
      Eigen::Vector3d p_global((Rfull*(Rguess*p_body + tguess) + tfull));
      point_out.x = p_global(0);
      point_out.y = p_global(1);
      point_out.z = p_global(2);
      point_out.intensity = point_src.intensity;
//      fout_dbg << "point_out " << point_out.x << " " << point_out.y << " " << point_out.z << std::endl;

      vector<float> pointSearchSqDis(NUM_MATCH_POINTS);
//      vector<int> pointIndices(NUM_MATCH_POINTS);

      PointVector points_near;

      /** Find the closest surfaces in the map **/
//      ikdtree_ptr->nearestKSearch(point_out, NUM_MATCH_POINTS, pointIndices, pointSearchSqDis);
      ikd_in->Nearest_Search(point_out, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
//      for (auto aidx:pointIndices){
//        points_near.push_back(cloud_submap->points[aidx]);
//        fout_dbg << cloud_submap->points[aidx].x << " " << cloud_submap->points[aidx].y << " " << cloud_submap->points[aidx].z << std::endl;
//      }
//      fout_dbg << "points_near.size()" << points_near.size();

//      for (auto adis:pointSearchSqDis){
//        fout_dbg << sqrt(adis) << " ";
//      }
      rmse += sqrt(pointSearchSqDis[0]);


      if (points_near.size() < NUM_MATCH_POINTS)
      {
        point_selected_surf[i] = false;
      }
      else
      {
        point_selected_surf[i] = pointSearchSqDis[NUM_MATCH_POINTS - 1] > 100 ? false : true;
      }

      if (!point_selected_surf[i]) continue;


      VF(4) pabcd; V3D acenter;
      point_selected_surf[i] = false;
      if (esti_plane(pabcd, points_near, 0.5f)) //(planeValid)
      {
//        for (auto aidx:pointIndices){
//          pointnear_tmp->push_back(cloud_submap->points[aidx]);
//        }
        acenter = ExtraLib::esti_center(points_near);
//        fout_dbg << "acenter" << acenter.transpose();

//        acenter = V3D(points_near[0].x, points_near[0].y, points_near[0].z);
        if ((p_global - acenter).norm() < 0.25) continue;

        normvec->points[i].x = acenter(0);
        normvec->points[i].y = acenter(1);
        normvec->points[i].z = acenter(2);
//        normvec_tmp->points[i].x = points_near[0].x;
//        normvec_tmp->points[i].y = points_near[0].y;
//        normvec_tmp->points[i].z = points_near[0].z;
        normvec->points[i].normal_x = pabcd(0);
        normvec->points[i].normal_y = pabcd(1);
        normvec->points[i].normal_z = pabcd(2);
        point_selected_surf[i] = true;
        effect_num++;
      }
    }
    rmse /= cloud_size;
    if (rmse < rmse_thr){
      elapsed_ms = 1000*(omp_get_wtime() - match_start_tmp);
      fout_dbg  << elapsed_ms << "ms" << "______all done______" <<endl;
      return true;
    }
    elapsed_ms = 1000*(omp_get_wtime() - match_start_tmp);
    fout_dbg  << elapsed_ms << "ms" << " done A b construction" <<endl;
//    fout_dbg<<"effect_num: " << effect_num << std::endl;
    Eigen::MatrixXd A(effect_num,6);
    Eigen::MatrixXd b(effect_num,1);
//    Eigen::MatrixXd diff(effect_num,3);
    effect_num = 0;
    for (int i = 0; i < cloud_size; i++)
    {
        if (point_selected_surf[i])
        {
          PointType point_src  = result_pplane->points[i];
          PointType point_tar = normvec->points[i];
//          fout_dbg << "point_src: " << point_src << std::endl;
//          fout_dbg << "point_tar: " << point_tar << std::endl;

          A(effect_num,0) = point_tar.normal_z*point_src.y-point_tar.normal_y*point_src.z;
          A(effect_num,1) = point_tar.normal_x*point_src.z-point_tar.normal_z*point_src.x;
          A(effect_num,2) = point_tar.normal_y*point_src.x-point_tar.normal_x*point_src.y;
          A(effect_num,3) = point_tar.normal_x;
          A(effect_num,4) = point_tar.normal_y;
          A(effect_num,5) = point_tar.normal_z;

          V3D dif = V3D(point_tar.x - point_src.x, point_tar.y - point_src.y,\
                                               point_tar.z - point_src.z);
//          fout_dbg << "dif: " << dif.transpose() << std::endl;
//          diff.row(effect_num) = dif;
          b(effect_num,0) = (dif.dot(V3D(point_tar.normal_x, point_tar.normal_y,\
                                              point_tar.normal_z)));

          effect_num ++;
//          fout_dbg<< "effect_num:" <<effect_num <<endl;
        }
    }
    elapsed_ms = 1000*(omp_get_wtime() - match_start_tmp);
    fout_dbg  << elapsed_ms << "ms" << " done point paris accumulation" <<endl;

//    fout_dbg< "b avg: " << sum(b)/float(effect_num) << std::endl;

    // solve x
    //Eigen::MatrixXd x = ((A.transpose()*A).inverse())*(A.transpose())*b;

    Eigen::MatrixXd A_inv = A.completeOrthogonalDecomposition().pseudoInverse();
    Eigen::MatrixXd x = A_inv*b;
    // rotation matrix
    double sin_a = sin(x(0));
    double cos_a = cos(x(0));
    double sin_b = sin(x(1));
    double cos_b = cos(x(1));
    double sin_y = sin(x(2));
    double cos_y = cos(x(2));
    M3D Rdelta;
    Rdelta(0,0) = cos_y*cos_b;
    Rdelta(0,1) = -sin_y*cos_a+cos_y*sin_b*sin_a;
    Rdelta(0,2) = sin_y*sin_a+cos_y*sin_b*cos_a;
    Rdelta(1,0) = sin_y*cos_b;
    Rdelta(1,1) = cos_y*cos_a+sin_y*sin_b*sin_a;
    Rdelta(1,2) = -cos_y*sin_a+sin_y*sin_b*cos_a;
    Rdelta(2,0) = -sin_b;
    Rdelta(2,1) = cos_b*sin_a;
    Rdelta(2,2) = cos_b*cos_a;

    V3D tdelta;
    // translation vector
    tdelta(0) = x(3);
    tdelta(1) = x(4);
    tdelta(2) = x(5);

    Rfull = Rdelta*Rfull;
    tfull = Rdelta*tfull + tdelta;

    fout_dbg << "Rfull: " << RotMtoEuler(Rfull).transpose() << endl;
    fout_dbg << "tfull: " << tfull.transpose() << endl;
    fout_dbg << "rmse: " << rmse << endl;
    elapsed_ms = 1000*(omp_get_wtime() - match_start_tmp);
    fout_dbg  << elapsed_ms << "ms" << "______one iteration done______" <<endl;
  }
  elapsed_ms = 1000*(omp_get_wtime() - match_start_tmp);
  fout_dbg  << elapsed_ms << "ms" << "______all done______" <<endl;

  if (rmse < rmse_thr)
    return true;

  return false;
}

void set_KF_pose(esekfom::esekf<state_ikfom, 12, input_ikfom> & kf, state_ikfom &tmp_state, const boost::shared_ptr<KD_TREE<PointType>> &ikd_in, \
               const PointCloudXYZI::Ptr &feats_down_body, const MD(4,4) &Tcomb, const MD(4,4) &Ticp, const MD(4,4) &Tcomb_nooff, const M3D &Rvel, const int pos_id_lc,\
               std_msgs::Float32MultiArrayPtr &notification_msg, std_msgs::Float64MultiArrayPtr &notification_msg2, std::ostream &fout_dbg)
{
    fout_dbg<<"--------------------set_KF_pos--------------------" <<endl;
    notification_msg->data.push_back(2);
    notification_msg->data.push_back(RotMtoEuler(tmp_state.rot.toRotationMatrix())[0]);
    notification_msg->data.push_back(RotMtoEuler(tmp_state.rot.toRotationMatrix())[1]);
    notification_msg->data.push_back(RotMtoEuler(tmp_state.rot.toRotationMatrix())[2]);
    notification_msg->data.push_back(tmp_state.pos[0]);
    notification_msg->data.push_back(tmp_state.pos[1]);
    notification_msg->data.push_back(tmp_state.pos[2]);
    notification_msg2->data.push_back(2);
    notification_msg2->data.push_back(RotMtoEuler(tmp_state.rot.toRotationMatrix())[0]);
    notification_msg2->data.push_back(RotMtoEuler(tmp_state.rot.toRotationMatrix())[1]);
    notification_msg2->data.push_back(RotMtoEuler(tmp_state.rot.toRotationMatrix())[2]);
    notification_msg2->data.push_back(tmp_state.pos[0]);
    notification_msg2->data.push_back(tmp_state.pos[1]);
    notification_msg2->data.push_back(tmp_state.pos[2]);
    fout_dbg << "KF pos before correction: " << kf.get_x().pos.transpose() << endl;

    tmp_state.rot = (Ticp*Tcomb_nooff).block<3,3>(0,0);
    tmp_state.pos = (Ticp*Tcomb_nooff).block<3,1>(0,3);
if (multisession_mode == 1)
{
    if (!first_correction_set) tmp_state.vel = Rvel*tmp_state.vel;
}
    kf.change_x(tmp_state);
    do_posecorrection = true;
    pos_replace = tmp_state.pos;
    rot_replace = tmp_state.rot;
    vel_replace = tmp_state.vel;

    notification_msg->data.push_back(RotMtoEuler(tmp_state.rot.toRotationMatrix())[0]);
    notification_msg->data.push_back(RotMtoEuler(tmp_state.rot.toRotationMatrix())[1]);
    notification_msg->data.push_back(RotMtoEuler(tmp_state.rot.toRotationMatrix())[2]);
    notification_msg->data.push_back(tmp_state.pos[0]);
    notification_msg->data.push_back(tmp_state.pos[1]);
    notification_msg->data.push_back(tmp_state.pos[2]);
    notification_msg2->data.push_back(RotMtoEuler(tmp_state.rot.toRotationMatrix())[0]);
    notification_msg2->data.push_back(RotMtoEuler(tmp_state.rot.toRotationMatrix())[1]);
    notification_msg2->data.push_back(RotMtoEuler(tmp_state.rot.toRotationMatrix())[2]);
    notification_msg2->data.push_back(tmp_state.pos[0]);
    notification_msg2->data.push_back(tmp_state.pos[1]);
    notification_msg2->data.push_back(tmp_state.pos[2]);
    fout_dbg << "KF pos after correction: " << kf.get_x().pos.transpose() << endl;

//    int cloud_size = feats_down_body->size();
////    assert(cloud_size < 100000);
//    M3D R = (Ticp*Tcomb).block<3,3>(0,0);
//    V3D t = (Ticp*Tcomb).block<3,1>(0,3);
//    float rmse = 0.0f;
//    for (int i = 0; i < cloud_size; i++)
//    {
//        const PointType point_body = feats_down_body->points[i];
//        PointType &point_world = feats_down_guess->points[i];
//        V3D p_body(point_body.x, point_body.y, point_body.z);
//        V3D p_global(R*p_body + t);
//        point_world.x = p_global(0);
//        point_world.y = p_global(1);
//        point_world.z = p_global(2);
//        point_world.intensity = point_body.intensity;

//        ikd_in->Nearest_Search(point_world, 1, points_near, pointSearchSqDis);
//        rmse += sqrt(pointSearchSqDis[0]);
//    }
//    fout_dbg<<endl<<"setpose rmse :"<<rmse/cloud_size<<endl;
}

