/*This code is the implementation of our paper "LTA-OM: Long-Term Association
LiDAR-Inertial Odometry and Mapping".

Current Developer: Zuhao Zou < zuhaozou@yahoo.com >

If you use any code of this repo in your academic research, please cite at least
one of our papers:
[1] Zou, Zuhao, et al. "LTA-OM: Long-Term Association LiDAR-Inertial
    Odometry and Mapping"
[2] Yuan, Chongjian, et al. "Std: Stable triangle descriptor for 3d place recognition"
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
#include "keyframe_containner.hpp"
#include "TunningPointPairsFactor.h"
#include "common_lib.hpp"

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/UInt64.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseWithCovariance.h>
#include <pcl/common/time.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/exceptions.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/slam/dataset.h>
#include <gtsam/nonlinear/Marginals.h>

#include <deque>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <mutex>
#include <thread>
#include <math.h>
#include <unordered_map>
#include <unistd.h>

#define PI 3.14159265

//#define as_node
#ifdef as_node
#include "loop_optimization_node.h"
#endif

//#define robust_estimator
#ifndef robust_estimator
#define marg
#endif
int multisession_mode = 0; //disabled by default
float overlap_score_thr = 0.5;
//float first_overlap_score_thr = 0.8;
bool dosave_pcd = false;
int pcdsave_step = 10;
bool dopub_corrmap = false;
int pairfactor_num = 6;
float plane_inliner_ratio_thr = 0.5;

float residual_thr = 2.0;
float vs_for_ovlap = 2.0;

std::string save_directory, pgo_scan_directory;
std::string poses_opt_fname, poses_raw_fname;
std::fstream poses_opt_file, poses_raw_file;//, poses_opt_all_file;
std::fstream times_opt_file;
std::fstream lc_file;
std::fstream opt_debug_file;
std::fstream time_full_pgo_thread;
ros::Publisher pubAndSaveGloablMapAftPGO, pubOdomAftPGO, pubPathAftPGO, \
               pubPathAftPGOPrior, pubLoopClosure;

std::deque<nav_msgs::Odometry::ConstPtr> odom_buf_;
std::deque<sensor_msgs::PointCloud2ConstPtr> cloud_buf_, interT_buf_, inter2T_buf_, LCT_buf_;
std::deque<geometry_msgs::PoseWithCovarianceConstPtr> loopclosure_buf_;

std::unordered_map<VOXEL_LOC, gtsam::Pose3> lc_pose_uomap;
//std::unordered_map<int, std::pair<gtsam::Pose3,gtsam::Pose3>> largejump_uomap;
std::unordered_map<VOXEL_LOC, std::pair<pcl::PointCloud<PointType>::Ptr, pcl::PointCloud<PointType>::Ptr>> cornerpairs_uomap;
std::unordered_map<int, bool> val_inserted;
std::unordered_map<int, bool> prior_inserted;


std::deque<KeyFrame> keyframes_;
std::deque<KeyFrame> keyframes_prior_;
int relocal_status = -1;

gtsam::FastVector<size_t> factors_toremove;
std::mutex mtx_sub_, mtx_pgo_, mtx_keyf_read_;

gtsam::NonlinearFactorGraph gts_graph_, gts_graph_recover_;
gtsam::ISAM2 *isam_;
gtsam::Values gts_init_vals_;
gtsam::Values gts_init_vals_recover_;
gtsam::Values gts_cur_vals_;
gtsam::ISAM2Params parameters;

gtsam::NonlinearFactorGraph refined_graph;
std::unordered_map<int, bool> factor_del;

gtsam::noiseModel::Diagonal::shared_ptr pose_start_noise_, pose_noise_, pose_noise2_;
gtsam::noiseModel::Base::shared_ptr loopclousure_noise_, point_noise_, lc_point_noise_;

int lc_curr_idx, lc_prev_idx, last_lc_prev_idx, last_lc_curr_idx;
float overlap_percen;
bool just_loop_closure = false;
int loop_corr_counter = 0;
bool wait_1more_loopclosure = false;

int fastlio_notify_type = 0;
gtsam::Pose3 subpose_beforejump, subpose_afterjump;
uint64_t jump_time = 0;

pcl::PointCloud<PointType>::Ptr currKeyframeCloud_icp(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr currKeyframeCloud_tmp(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr targetKeyframeCloud_tmp(new pcl::PointCloud<PointType>());

double elapsed_ms_opt = 0.0f;
int opt_count = 0;
double elapsed_ms_max = 0.0f;
float passed_dis = 0.0f;
float passed_dis_atlast_lc = 0.0f;

int graphpose_count = 0;
bool has_add_prior_node = false;
float x_range_max, y_range_max, z_range_max;
float x_range_min, y_range_min, z_range_min;

bool loadConfig(ros::NodeHandle &nh)
{
    if(nh.hasParam("OverlapScoreThr"))
        ROS_INFO("Loop Correction Node Successfully Loaded Config Parameters.");
    else
    {
        ROS_ERROR("Loop Correction Node Cannot Find Config Parameters File!");
        return false;
    }
    nh.param<float>("OverlapScoreThr",overlap_score_thr,0.5);
    //nh.param<float>("FirstOverlapScoreThr",first_overlap_score_thr,0.8);
    nh.param<int>("NumPrKeyPtFactor",pairfactor_num,6);
    nh.param<float>("PlaneInlinerRatioThr",plane_inliner_ratio_thr,0.5);
    nh.param<float>("VoxelSizeForOverlapCalc",vs_for_ovlap,2);
    nh.param<bool>("SavePCD",dosave_pcd,false);
    nh.param<int>("PCDSaveStep",pcdsave_step,10);
    nh.param<bool>("PubCorrectedMap",dopub_corrmap,false);
    nh.param<string>("SaveDir",save_directory,"");
    nh.param<int>("multisession_mode",multisession_mode, 0);

    std::vector<double> cov1, cov2, cov3;
    nh.param<vector<double>>("AdjKPFCov", cov1, vector<double>());
    nh.param<vector<double>>("LCKPFCov", cov2, vector<double>());
    nh.param<vector<double>>("MargFCov", cov3, vector<double>());

    gtsam::Vector noise_vec3(3);
    noise_vec3 << cov1[0], cov1[1], cov1[2];
    point_noise_ = gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::Cauchy::Create(1),
        gtsam::noiseModel::Diagonal::Variances(noise_vec3));
    noise_vec3 << cov2[0], cov2[1], cov2[2];
    lc_point_noise_ = gtsam::noiseModel::Diagonal::Variances(noise_vec3);
    pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);

    gtsam::Vector noise_vec6(6);
    noise_vec6 << cov3[0], cov3[1], cov3[2], cov3[3], cov3[4], cov3[5];
    pose_noise2_ = gtsam::noiseModel::Diagonal::Variances(noise_vec6);
    return true;
}

int wrap4Containner(int val)
{
    int out = val<0?-val-1:val;
    return out;
}
int wrap4Gtsam(int val)
{
    int out = val<0?-val+50000:val;
    return out;
}
void odometryCallback(const nav_msgs::Odometry::ConstPtr &odom_msg)
{
    mtx_sub_.lock();
    odom_buf_.push_back(odom_msg);
    mtx_sub_.unlock();
}

void notificationCallback(const std_msgs::Float64MultiArray::ConstPtr &msg)
{
    if (msg->data[0] == 2 && fastlio_notify_type == 1)
    {
        subpose_beforejump = gtsam::Pose3(gtsam::Rot3::RzRyRx(msg->data[1], msg->data[2], msg->data[3]), \
            gtsam::Point3(msg->data[4], msg->data[5], msg->data[6]));
        subpose_afterjump = gtsam::Pose3(gtsam::Rot3::RzRyRx(msg->data[7], msg->data[8], msg->data[9]), \
            gtsam::Point3(msg->data[10], msg->data[11], msg->data[12]));
        opt_debug_file << "*subpose_afterjump: " << subpose_afterjump.translation()  <<endl;
    }
    fastlio_notify_type = int(msg->data[0]);
}
void jumptimeCallback(const std_msgs::UInt64::ConstPtr &msg)
{
    jump_time = msg->data;
}
void cloudCallback(const sensor_msgs::PointCloud2ConstPtr &cloud_msg)
{
    mtx_sub_.lock();
    cloud_buf_.push_back(cloud_msg);
    mtx_sub_.unlock();
}
void interKPPairCallback(const sensor_msgs::PointCloud2ConstPtr &cloud_msg)
{
    mtx_sub_.lock();
    interT_buf_.push_back(cloud_msg);
    mtx_sub_.unlock();
}
void inter2KPPairCallback(const sensor_msgs::PointCloud2ConstPtr &cloud_msg)
{
    mtx_sub_.lock();
    inter2T_buf_.push_back(cloud_msg);
    mtx_sub_.unlock();
}
void loopClosureKPPairCallback(const sensor_msgs::PointCloud2ConstPtr &cloud_msg)
{
    mtx_sub_.lock();
    LCT_buf_.push_back(cloud_msg);
    mtx_sub_.unlock();
}
void removeLC(const int curr_node_idx, const int prev_node_idx)
{
    int l = 0;
    while (l < LCT_buf_.size())             // remove rejected LC pairs from LC KP messgage buffer
    {
        auto LCTmsg = LCT_buf_[l];
        pcl::PointCloud<PointType>::Ptr cloud_tmp(new pcl::PointCloud<PointType>());
        pcl::fromROSMsg(*(LCTmsg), *cloud_tmp);
        int prev_idx_lct = cloud_tmp->points[0].intensity;
        int curr_idx_lct = cloud_tmp->points[1].intensity;
        if (curr_idx_lct == curr_node_idx && prev_idx_lct == prev_node_idx)
            break;
        l++;
    }
    if (l < LCT_buf_.size()) LCT_buf_.erase(LCT_buf_.begin()+l);
}
void loopClosureCallback(const geometry_msgs::PoseWithCovarianceConstPtr& lc_msg)
{
    mtx_sub_.lock();
    loopclosure_buf_.push_back(lc_msg);
    mtx_sub_.unlock();
}
bool keyPointDisSort(std::tuple<gtsam::Vector3,gtsam::Vector3,float> a, std::tuple<gtsam::Vector3,gtsam::Vector3,float> b) { return (std::get<2>(a) < std::get<2>(b)); }

pcl::PointCloud<PointType>::Ptr currKeyframeCloud_ds(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr targetKeyframeCloud_ds(new pcl::PointCloud<PointType>());
float calculateOverlapScore(const pcl::PointCloud<PointType>::Ptr& currKeyframeCloud, const pcl::PointCloud<PointType>::Ptr& targetKeyframeCloud,
                            const Eigen::Matrix4f& T_cur, const Eigen::Matrix4f& T_prev, const Eigen::Matrix4f& guess,
                            const float &voxel_box_size, const int &num_pt_max, const int pre_idx, const int cur_idx)
{
    opt_debug_file << "targetKeyframeCloud size " << targetKeyframeCloud->size() << std::endl;
    opt_debug_file << "currKeyframeCloud size " << currKeyframeCloud->size() << std::endl;

    pcl::VoxelGrid<PointType> sor;
    sor.setInputCloud(currKeyframeCloud);
    sor.setLeafSize(0.5*voxel_box_size, 0.5*voxel_box_size, 0.5*voxel_box_size);
    sor.filter(*currKeyframeCloud_ds);

    sor.setInputCloud(targetKeyframeCloud);
    sor.setLeafSize(0.5*voxel_box_size, 0.5*voxel_box_size, 0.5*voxel_box_size);
    sor.filter(*targetKeyframeCloud_ds);

    int num_pt_cur = int(currKeyframeCloud_ds->size());
    std::vector<int> indices;
    int sample_gap;
    if (num_pt_cur > 2*num_pt_max)
        sample_gap = ceil(double(num_pt_cur)/double(num_pt_max));
    else
        sample_gap = 1;
    for (int i = 0; i < num_pt_cur; i+=sample_gap)  // downsample
    {
        indices.push_back(i);
    }


    int num_pt_target = int(targetKeyframeCloud_ds->size());
    float cloud_size_ratio = std::min(float(num_pt_cur) / float(num_pt_target), float(num_pt_target) / float(num_pt_cur));
    opt_debug_file << "[FPR]: submap pair's pt numbers ratio: " << cloud_size_ratio << std::endl;

    pcl::PointCloud<PointType>::Ptr cloud_in_transed (new pcl::PointCloud<PointType>);
    pcl::transformPointCloud(*currKeyframeCloud_ds, indices, *cloud_in_transed, T_prev*guess*T_cur.inverse(), false);

    std::unordered_map<VOXEL_LOC, int> uomp_3d;
    CutVoxel3d(uomp_3d, targetKeyframeCloud_ds, voxel_box_size);  // cut voxel for counting hit

    int count1 = 0;
    int count2 = 0;
    for (int i = 0; i < cloud_in_transed->size(); i++)
    {
        auto &a_pt = cloud_in_transed->points[i];
        Eigen::Vector3f pt(a_pt.x, a_pt.y, a_pt.z);
        float loc_xyz[3];
        for(int j = 0; j < 3; j++)
        {
            loc_xyz[j] = pt[j] / voxel_box_size;
            if(loc_xyz[j] < 0)
            {
                loc_xyz[j] -= 1.0;
            }
        }
        VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
        auto iter = uomp_3d.find(position);
        if(iter != uomp_3d.end())
        {
            if (iter->second == 0)
                count1++;
            iter->second++;
            count2++;
        }
    }

    float score = (float(count2)/float(cloud_in_transed->size()))*cloud_size_ratio;
//  float score = float(count2)/float(cloud_in_transed->size())<float(count1)/float(uomp_3d.size())?float(count2)/float(cloud_in_transed->size()):float(count1)/float(uomp_3d.size());
    if (score > overlap_score_thr)
    {
        bool is_plane = CheckIfJustPlane(cloud_in_transed, plane_inliner_ratio_thr);
        if (is_plane)
        {
            opt_debug_file <<  "[FPR]: reject, the target cloud is like a plane." << std::endl;
            return 0.0f;
        }
    }
    if(dosave_pcd)
    {
        if (score > overlap_score_thr)
        {
            currKeyframeCloud_tmp = cloud_in_transed;
            targetKeyframeCloud_tmp = targetKeyframeCloud_ds;

            try
            {
                pcl::io::savePCDFileBinary(pgo_scan_directory + "/LargeOverlap/" + std::to_string(pre_idx) + "_" + std::to_string(cur_idx) + "_" + std::to_string(score*100) + "%_cloud_cur_guess.pcd", *cloud_in_transed); // scan
                pcl::io::savePCDFileBinary(pgo_scan_directory + "/LargeOverlap/" + std::to_string(pre_idx) + "_" + std::to_string(cur_idx) + "cloud_target.pcd", *targetKeyframeCloud_ds); // scan
            } catch (pcl::PCLException e)
            {
                ROS_ERROR("%s", e.what());
            }

        }
        else
        {
            try
            {
                pcl::io::savePCDFileBinary(pgo_scan_directory + "/LowOverlap/" + std::to_string(pre_idx) + "_" + std::to_string(cur_idx) + "_" + std::to_string(score*100) + "%_cloud_cur_guess.pcd", *cloud_in_transed); // scan
                pcl::io::savePCDFileBinary(pgo_scan_directory + "/LowOverlap/" + std::to_string(pre_idx) + "_" + std::to_string(cur_idx) + "cloud_target.pcd", *targetKeyframeCloud_ds); // scan
            } catch (pcl::PCLException e)
            {
                ROS_ERROR("%s", e.what());
            }
        }
    }

    return score;
//  return (float(count1)/float(uomp_3d.size())) * (float(count2)/float(cloud_in_transed->size()));
}

void pubPath()
{
    if (keyframes_.empty())  return;
//  if (loop_corr_counter!=2) return;
    nav_msgs::Odometry odomAftPGO;
    nav_msgs::PathPtr pathAftPGO (new nav_msgs::Path());
    pathAftPGO->header.frame_id = "/camera_init"; //"/world";
    for (int i=0; i < keyframes_.size(); i++)
    {
        nav_msgs::Odometry odomAftPGOthis;
        odomAftPGOthis.header.frame_id = "/camera_init"; //"/world";
        odomAftPGOthis.child_frame_id = "/aft_pgo";
        odomAftPGOthis.header.stamp = keyframes_[i].KeyTime;
        odomAftPGOthis.header.seq = i;
        odomAftPGOthis.pose.pose.position.x = keyframes_[i].KeyPoseOpt.x;
        odomAftPGOthis.pose.pose.position.y = keyframes_[i].KeyPoseOpt.y;
        odomAftPGOthis.pose.pose.position.z = keyframes_[i].KeyPoseOpt.z;

        odomAftPGOthis.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(keyframes_[i].KeyPoseOpt.roll, keyframes_[i].KeyPoseOpt.pitch, keyframes_[i].KeyPoseOpt.yaw);

        geometry_msgs::PoseStamped poseStampAftPGO;
        poseStampAftPGO.header = odomAftPGOthis.header;
        poseStampAftPGO.pose = odomAftPGOthis.pose.pose;

        pathAftPGO->header.stamp = odomAftPGOthis.header.stamp;
        pathAftPGO->header.frame_id = "/camera_init"; //"/world";
        pathAftPGO->poses.push_back(poseStampAftPGO);
    }
    pubPathAftPGO.publish(pathAftPGO);
    ros::spinOnce();
    opt_debug_file << "[PubPath]: path size " << pathAftPGO->poses.size() << std::endl;
}

visualization_msgs::Marker lines_;
void pubLoopClosureVisualization(const int prev_idx, const int curr_idx){
    lines_.header.stamp = ros::Time::now();
    lines_.header.frame_id = "/camera_init"; //"/world";
    lines_.ns = "loopclosure_lines";
    lines_.action = visualization_msgs::Marker::ADD;
    lines_.pose.orientation.w = 1.0f;
    lines_.id = loop_corr_counter;
    lines_.type = visualization_msgs::Marker::LINE_LIST;
    lines_.scale.x = 0.5;
    lines_.color.r = 1.0f;
    lines_.color.g = 0.0f;
    lines_.color.b = 0.0f;
    lines_.color.a = 1.0f;
    geometry_msgs::Point point_a, point_b;

if (multisession_mode == 1)
{
    bool on_prior = prev_idx < 0;
    point_a.x = on_prior?keyframes_prior_[wrap4Containner(prev_idx)].KeyPoseOpt.x:keyframes_[prev_idx].KeyPose.x;
    point_a.y = on_prior?keyframes_prior_[wrap4Containner(prev_idx)].KeyPoseOpt.y:keyframes_[prev_idx].KeyPose.y;
    point_a.z = on_prior?keyframes_prior_[wrap4Containner(prev_idx)].KeyPoseOpt.z:keyframes_[prev_idx].KeyPose.z;

    point_b.x = keyframes_[curr_idx].KeyPose.x;
    point_b.y = keyframes_[curr_idx].KeyPose.y;
    point_b.z = keyframes_[curr_idx].KeyPose.z;
}
else
{
    point_a.x = keyframes_[prev_idx].KeyPose.x;
    point_a.y = keyframes_[prev_idx].KeyPose.y;
    point_a.z = keyframes_[prev_idx].KeyPose.z;

    point_b.x = keyframes_[curr_idx].KeyPose.x;
    point_b.y = keyframes_[curr_idx].KeyPose.y;
    point_b.z = keyframes_[curr_idx].KeyPose.z;
}

    lines_.points.push_back(point_a);
    lines_.points.push_back(point_b);

    if (pubLoopClosure.getNumSubscribers() != 0)
    {
        pubLoopClosure.publish(lines_);
    }
}

void addInitialPoseValues(int pose_index, gtsam::Pose3 pose_prior)
{
    int pose_index_wrap = wrap4Gtsam(pose_index);
    if (val_inserted.find(pose_index_wrap) != val_inserted.end())
    {
        if (val_inserted[pose_index_wrap] == true)
            return;
    }
    gts_init_vals_.insert(pose_index_wrap, pose_prior);
    gts_init_vals_recover_.insert(pose_index_wrap, pose_prior);
    val_inserted[pose_index_wrap] = true;
}

void addPriorPoses(int pose_index, gtsam::Pose3 pose_prior)
{
    if (prior_inserted.find(pose_index) != prior_inserted.end())
    {
        if (prior_inserted[pose_index] == true)
            return;
    }
    gts_graph_.add(gtsam::PriorFactor<gtsam::Pose3>(pose_index, pose_prior, pose_start_noise_));
    gts_graph_recover_.add(gtsam::PriorFactor<gtsam::Pose3>(pose_index, pose_prior, pose_start_noise_));
    prior_inserted[pose_index] = true;
    addInitialPoseValues(pose_index, pose_prior);
    opt_debug_file <<  "[Add Prior Factor]: prior at " << pose_index << std::endl;
    std::cout      <<  "[Add Prior Factor]: prior at " << pose_index << std::endl;
}

void addPriorPosesFromPriorInfo(int prior_pose_index, int online_pose_idx)
{
    if (prior_inserted.find(online_pose_idx) != prior_inserted.end())
    {
        if (prior_inserted[online_pose_idx] == true)
            return;
    }

    gtsam::Pose3 prior_keyposeopt = Pose6DToGTSPose(keyframes_prior_[wrap4Containner(prior_pose_index)].KeyPoseOpt);
    VOXEL_LOC position2; position2.x = (int)(prior_pose_index), position2.y = (int)(online_pose_idx), position2.z = (int)(0);
    if (lc_pose_uomap.find(position2) != lc_pose_uomap.end())
    {
        gtsam::Pose3 brh_T_bc = lc_pose_uomap[position2];
        gtsam::Pose3 pose_online = prior_keyposeopt*brh_T_bc;
        gts_graph_.add(gtsam::PriorFactor<gtsam::Pose3>(online_pose_idx, pose_online, pose_start_noise_));
        prior_inserted[online_pose_idx] = true;
        opt_debug_file <<  "[Add Prior Factor]: prior at " << online_pose_idx << std::endl;
        std::cout      <<  "[Add Prior Factor]: prior at " << online_pose_idx << std::endl;
    }
}

void loadPriorMapAndPoses()
{
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr datacloud(new pcl::PointCloud<pcl::PointXYZINormal>());
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr tmpmap(new pcl::PointCloud<pcl::PointXYZINormal>());
    std::string pcd_name = save_directory + "map_prior/clouds_corrected.pcd";
    opt_debug_file << "[LoadPrior]: Loading pcd from " << pcd_name << std::endl;
    std::cout      << "[LoadPrior]: Loading pcd from " << pcd_name << std::endl;
    if (pcl::io::loadPCDFile(pcd_name, *datacloud) == -1)
    {
        std::cerr << "Failed to load " << pcd_name << std::endl;
        return;
    }
//    pcl::io::loadPCDFile(pcd_name, *datacloud);
    opt_debug_file << "[LoadPrior]: Extracted KF "  << std::endl;
    std::cout      << "[LoadPrior]: Extracted KF "  << std::endl;
    std::vector<Eigen::Vector2i> lc_pairs;
    for(int j = 0; j < datacloud->points.size(); j++)
    {
        auto a_pt = datacloud->points[j];
        if(fabs(a_pt.x - (-1010.1)) < 0.01)
        {
            int cloud_size = a_pt.z;
            if (cloud_size != 0)
            {
                j += cloud_size;
            }
            Pose6D a_pose;
            KeyFrame akeyframe;

            j++;
            a_pt = datacloud->points[j];
            a_pose.roll = a_pt.x; a_pose.pitch = a_pt.y; a_pose.yaw = a_pt.z;
            j++;
            a_pt = datacloud->points[j];
            a_pose.x = a_pt.x; a_pose.y = a_pt.y; a_pose.z = a_pt.z;
            akeyframe.KeyPoseOpt = a_pose;
            akeyframe.KeyPoseCompare = a_pose;

            j++;
            a_pt = datacloud->points[j];
            a_pose.roll = a_pt.x; a_pose.pitch = a_pt.y; a_pose.yaw = a_pt.z;
            j++;
            a_pt = datacloud->points[j];
            a_pose.x = a_pt.x; a_pose.y = a_pt.y; a_pose.z = a_pt.z;

            akeyframe.KeyPose = a_pose;
            akeyframe.KeyCloud = tmpmap;
            pcl::PointCloud<pcl::PointXYZINormal>::Ptr empty_cloud (new pcl::PointCloud<pcl::PointXYZINormal>());
            tmpmap = empty_cloud;
            keyframes_prior_.push_back(akeyframe);
            opt_debug_file << " " << keyframes_prior_.size() ;
            std::cout      << " " << keyframes_prior_.size() ;
            continue;
        }
        if(fabs(a_pt.x - (-2020.2)) < 0.01)
        {
            lc_pairs.push_back(Eigen::Vector2i(a_pt.y, a_pt.z));
            continue;
        }
        tmpmap->push_back(a_pt);
    }
    opt_debug_file << std::endl << "[LoadPrior]: Loading prior cloud and poses done "  << std::endl;
    std::cout      << std::endl << "[LoadPrior]: Loading prior cloud and poses done "  << std::endl;
}

int last_kfsize = 0;
void correctPosesAndSaveTxt()
{
    if (last_kfsize==keyframes_.size()) return;
    last_kfsize = keyframes_.size();
    auto start1 = std::chrono::system_clock::now();
    poses_opt_file.open(poses_opt_fname, ios::out | ios::trunc);
    auto end1 = std::chrono::system_clock::now();
    auto elapsed_ms = (std::chrono::duration<double,std::milli>(end1 - start1)).count();
    for (int i = 0; i < keyframes_.size(); i++)
    {
        const gtsam::Pose3& pose_optimized = gts_cur_vals_.at<gtsam::Pose3>(gtsam::Symbol(i));
        tf2::Quaternion q;
        q.setRPY(pose_optimized.rotation().roll(), pose_optimized.rotation().pitch(), pose_optimized.rotation().yaw());
        auto start2 = std::chrono::system_clock::now();
        poses_opt_file     <<  keyframes_[i].KeyTime << " " << pose_optimized.x() << " " << pose_optimized.y() << " " << pose_optimized.z()
            << " " << q.getX() << " " << q.getY() << " " << q.getZ() << " " << q.getW() << std::endl;
        auto end2 = std::chrono::system_clock::now();
        elapsed_ms += (std::chrono::duration<double,std::milli>(end2 - start2)).count();
//        poses_opt_all_file <<  keyframes_[i].KeyTime << " " << pose_optimized.x() << " " << pose_optimized.y() << " " << pose_optimized.z()
//            << " " << q.getX() << " " << q.getY() << " " << q.getZ() << " " << q.getW() << " "      ;
        keyframes_[i].KeyPoseOpt = GTSPoseToPose6D(pose_optimized);
        keyframes_[i].pose_opt_set = true;
    }
//    poses_opt_all_file << std::endl;
    auto start3 = std::chrono::system_clock::now();
    poses_opt_file.close();
    auto end3 = std::chrono::system_clock::now();
    elapsed_ms += (std::chrono::duration<double,std::milli>(end3 - start3)).count();
    opt_debug_file << "save pose txt takes " << elapsed_ms << std::endl;
}

void printGraph(const gtsam::NonlinearFactorGraph& graph)
{
    opt_debug_file << "[PrintGraph]: ";
    int btw_count = 0; int tri_count = 0; int prior_count = 0;
    for(const boost::shared_ptr<gtsam::NonlinearFactor>& factor: graph) {
        boost::shared_ptr<gtsam::BetweenFactor<gtsam::Pose3> > pose3Between =
            boost::dynamic_pointer_cast<gtsam::BetweenFactor<gtsam::Pose3> >(factor);
        if (pose3Between)
        {
            gtsam::Key key1, key2;
            key1 = pose3Between->key1();
            key2 = pose3Between->key2();
            opt_debug_file << "btw(" << key1<<", " <<key2 <<")" << " ";
            btw_count++;
        }
        boost::shared_ptr<gtsam::TunningPointPairsFactor<gtsam::Pose3> > triBetween =
            boost::dynamic_pointer_cast<gtsam::TunningPointPairsFactor<gtsam::Pose3> >(factor);
        if (triBetween)
        {
            gtsam::Key key1, key2;
            key1 = triBetween->key1();
            key2 = triBetween->key2();
            opt_debug_file << "tri:(" << key1<<", " <<key2 <<")" << " ";
            tri_count++;
        }
        boost::shared_ptr<gtsam::PriorFactor<gtsam::Pose3>> priorPose =
            boost::dynamic_pointer_cast<gtsam::PriorFactor<gtsam::Pose3>>(factor);
        if (priorPose)
        {
            gtsam::Key key;
            key = priorPose->key();
            opt_debug_file << "prior:(" << key <<")" << " ";
            prior_count++;
        }
    }
    opt_debug_file << "btw: " << btw_count << " tri: " << tri_count << " prior: " << prior_count << std::endl;
}

bool checkResiduals(const float res_thr){
    std::unordered_map<VOXEL_LOC, bool> tmp;
    bool outflag = false;
    gtsam::NonlinearFactorGraph graph_cur = isam_->getFactorsUnsafe();

//    // Directly get residuals from gtsam graph, TODO::why residual not correct?
//    gts_cur_vals_ = isam_->calculateEstimate();
//    std::vector<std::pair<int,float>> residuals = graph_cur.getErrors(gts_cur_vals_);
//    for (auto res:residuals){
//        if (res.second > res_thr){
//            opt_debug_file << "(" << graph_cur[res.first]->keys()[0] << " " << graph_cur[res.first]->keys()[1] << "=" << res.second << ")";
//            outflag = true;
//        }
//    }
//    opt_debug_file  << std::endl;

    for(const boost::shared_ptr<gtsam::NonlinearFactor>& factor: graph_cur)   // Iterate over currect factor in the graph
    {
        boost::shared_ptr<gtsam::TunningPointPairsFactor<gtsam::Pose3> > triBetween =
            boost::dynamic_pointer_cast<gtsam::TunningPointPairsFactor<gtsam::Pose3> >(factor);
        gtsam::Key key1, key2;
        if (triBetween)                                                       // If it is a Adjacent KPF
        {
            key1 = triBetween->key1();
            key2 = triBetween->key2();
        }
        else
        {
//            boost::shared_ptr<gtsam::BetweenFactor<gtsam::Pose3> > odomBetween =
//                boost::dynamic_pointer_cast<gtsam::BetweenFactor<gtsam::Pose3> >(factor);
//            if (odomBetween){
//              key1 = odomBetween->key1();
//              key2 = odomBetween->key2();
//            }
//            else continue;
            continue;
        }
        gtsam::Pose3 T_prev;

if (multisession_mode == 1)
{
        if (key1 >= 50000)                                                      // We use key > 50000 for LC factor on prior
            T_prev = Pose6DToGTSPose(keyframes_prior_[key1-50001].KeyPoseOpt);  // "key1-50001" is the conversion to containner index
        else
            T_prev = Pose6DToGTSPose(keyframes_[key1].KeyPoseOpt);
}
else
{
        T_prev = Pose6DToGTSPose(keyframes_[key1].KeyPoseOpt);
}
        gtsam::Pose3 T_curr = Pose6DToGTSPose(keyframes_[key2].KeyPoseOpt);
        const gtsam::Pose3 _2T1_ = T_curr.between(T_prev);
        VOXEL_LOC position;
if (multisession_mode == 1)
{
        position.x = int(key1>=50000?-(key1-50000):key1),position.y = (int)(key2),position.z = (int)(0);
}
else
{
        position.x = (int)(key1),position.y = (int)(key2),position.z = (int)(0);
}
        if (tmp.find(position) != tmp.end()) continue;
        if (cornerpairs_uomap.find(position) == cornerpairs_uomap.end()) continue;
        pcl::PointCloud<PointType>::Ptr cloud_prev = cornerpairs_uomap[position].first;
        pcl::PointCloud<PointType>::Ptr cloud_curr = cornerpairs_uomap[position].second;
        if (cloud_prev->empty() || cloud_curr->empty())continue;

        tmp[position] = true;
        pcl::PointCloud<PointType>::Ptr cloud_prev_tran (new pcl::PointCloud<PointType>);
        pcl::transformPointCloud(*(cloud_prev), *cloud_prev_tran, GTSPoseToEigenM4f(_2T1_));
        int pairs_size = int(cloud_prev_tran->size());
        float residual_max = 0;
        for (int j = 0; j < pairs_size; j++)
        {
            const gtsam::Point3 q(cloud_prev_tran->points[j].x, cloud_prev_tran->points[j].y, cloud_prev_tran->points[j].z);
            const gtsam::Point3 p2(cloud_curr->points[j].x, cloud_curr->points[j].y, cloud_curr->points[j].z);
            float residual = (q - p2).norm();
            residual_max = residual > residual_max ? residual:residual_max;
        }
        opt_debug_file << "(" << key1 << " " << key2 << "=" << residual_max << ")";
        if (residual_max > res_thr)
        {
            opt_debug_file << "!!!!" ;
            outflag = true;
//            return outflag; // can uncomment this line for efficiency
        }
    }
    return outflag;
}

void marginalizeKPPairFactor(const int lc_start, const int lc_end, const bool to_print)  // Marginalize KPF for efficiency when some correct LC are accepted
{
    if (to_print) opt_debug_file << "[MargKPF]: ";
    gtsam::NonlinearFactorGraph graph_cur = isam_->getFactorsUnsafe();
    std::unordered_map<VOXEL_LOC, bool> tmp;
    if (to_print) opt_debug_file << " factors_toremove:";
    for (int i = 0; i < graph_cur.size(); i++)
    {
        if(factor_del.find(i) != factor_del.end()) continue;
        auto afactor_keys = graph_cur[i]->keys();
        if (afactor_keys.size() == 2)
        {
            if (afactor_keys[0] >= lc_start && afactor_keys[1] <= lc_end && afactor_keys[1] - afactor_keys[0] <= 2)  // Adj KPF
            {
                factors_toremove.push_back(i);
                if (to_print) opt_debug_file << "("  << afactor_keys[0] <<"," <<afactor_keys[1] << ")";
                factor_del[i] = true;
            }
            if (afactor_keys[0] != lc_start && afactor_keys[1] != lc_end && afactor_keys[1] - afactor_keys[0] > 2)   // LC KPF
            {
                factors_toremove.push_back(i);
                if (to_print) opt_debug_file << "("  << afactor_keys[0] <<"," <<afactor_keys[1] << ")";
                factor_del[i] = true;
                VOXEL_LOC position; position.x = (int)(afactor_keys[0]),position.y = (int)(afactor_keys[1]),position.z = (int)(0);
                if (tmp.find(position) == tmp.end())
                    tmp[position] = true;
            }
        }
    }
    if (to_print) opt_debug_file << std::endl;
    if (to_print) opt_debug_file << " refined_graph+: ";
    for (int i = lc_start; i < lc_end; i++)
    {
        const gtsam::Value& estimation_last = isam_->calculateEstimate(i);
        const gtsam::Value& estimation_curr = isam_->calculateEstimate(i+1);
        const gtsam::Pose3& pose_curr = estimation_curr.cast<gtsam::Pose3>();
        const gtsam::Pose3& pose_last = estimation_last.cast<gtsam::Pose3>();
        refined_graph.add(gtsam::BetweenFactor<gtsam::Pose3>(i, i+1, pose_last.between(pose_curr), pose_noise_));
    }
    for (auto iter = tmp.begin(); iter != tmp.end(); ++iter)
    {
        if (to_print) opt_debug_file << "(" << iter->first.x << "," <<iter->first.y << ")";
        const gtsam::Value& estimation_last = isam_->calculateEstimate(iter->first.x);
        const gtsam::Value& estimation_curr = isam_->calculateEstimate(iter->first.y);
        const gtsam::Pose3& pose_last = estimation_last.cast<gtsam::Pose3>();
        const gtsam::Pose3& pose_curr = estimation_curr.cast<gtsam::Pose3>();
        refined_graph.add(gtsam::BetweenFactor<gtsam::Pose3>(iter->first.x, iter->first.y, pose_last.between(pose_curr), pose_noise2_));
    }
    if (to_print) opt_debug_file << std::endl;
    gts_graph_ = refined_graph;
    gts_graph_recover_ = refined_graph;
    refined_graph.resize(0);
}

void optimize()
{
    if (gts_init_vals_.empty() || gts_graph_.empty())
        return;

    std::cout      << "[Optimize]: running isam2 optimization ..." << endl;
    opt_debug_file << "[Optimize]: running isam2 optimization ..." << endl;
    auto start1 = std::chrono::system_clock::now();
//    printGraph(gts_graph_);
    isam_->backup();         // self-defined isam function, you need to compile gtsam with provided note in Readme
#ifdef marg
    if (!factors_toremove.empty() &&just_loop_closure)
    {
        isam_->update(gts_graph_, gts_init_vals_, factors_toremove);
    }
    else
#endif
        isam_->update(gts_graph_, gts_init_vals_); // if isam->update() triggers segmentation fault, try parameters.factorization = gtsam::ISAM2Params::QR;

    auto isam_result = isam_->update();
    if (just_loop_closure)
    {
        isam_result = isam_->update();
        isam_result = isam_->update();
        isam_result = isam_->update();                  //result more stable if repeat
    }
    auto end1 = std::chrono::system_clock::now();
    auto elapsed_ms = (std::chrono::duration<double,std::milli>(end1 - start1)).count();

    gts_cur_vals_ = isam_->calculateEstimate();
    correctPosesAndSaveTxt();

//  pubPath();
    if (!just_loop_closure)
    {
        gts_graph_.resize(0);
        gts_graph_recover_.resize(0);
        gts_init_vals_.clear();
        gts_init_vals_recover_.clear();
        return;
    }

    elapsed_ms_opt += elapsed_ms;
    opt_count++;
    elapsed_ms_max = elapsed_ms>elapsed_ms_max?elapsed_ms:elapsed_ms_max;
    opt_debug_file <<  "[Optimize]: optimization 1 takes " << elapsed_ms << "ms" << std::endl;
    opt_debug_file <<  "[Optimize]: optimization with lc: -----------" << lc_prev_idx << " " << lc_curr_idx << std::endl;
    opt_debug_file <<  "[Optimize]: optimization avg takes " << elapsed_ms_opt/opt_count << "ms" << std::endl;
    opt_debug_file <<  "[Optimize]: optimization max takes " << elapsed_ms_max << "ms" << std::endl;

    bool bad_loopclosure = checkResiduals(residual_thr);
    times_opt_file << elapsed_ms << " " << !bad_loopclosure << std::endl;

    if (bad_loopclosure)
    {
        opt_debug_file <<  "[FPR]: reject, large residual appear." << std::endl;
//        opt_debug_file << "origin isam values size = " << isam_->calculateEstimate().size() << std::endl;
        isam_->recover();    // self-defined isam function, you need to compile gtsam with provided note in Readme

//        complete_graph = isam_->getFactorsUnsafe();
//        opt_debug_file << "aft recovered: " << std::endl;
//        printGraph(complete_graph);
if (multisession_mode == 1)
{
        if (relocal_status == 0)
        {
            relocal_status = -1;
            gts_graph_ = gts_graph_recover_;
            gts_init_vals_ = gts_init_vals_recover_;
        }
        else
        {
            isam_->update(gts_graph_recover_, gts_init_vals_);
            gts_cur_vals_ = isam_->calculateEstimate();
            correctPosesAndSaveTxt();

            gts_graph_.resize(0);
            gts_graph_recover_.resize(0);
            gts_init_vals_.clear();
            gts_init_vals_recover_.clear();
        }
}
else
{
        isam_->update(gts_graph_recover_, gts_init_vals_);
        gts_cur_vals_ = isam_->calculateEstimate();
        correctPosesAndSaveTxt();

        gts_graph_.resize(0);
        gts_graph_recover_.resize(0);
        gts_init_vals_.clear();
        gts_init_vals_recover_.clear();
}

        if (lc_pose_uomap.size() == 2) //this case is that first two lc not consistent
            lc_pose_uomap.clear();

        if(dosave_pcd)
        {
            try
            {
                pcl::io::savePCDFileBinary(pgo_scan_directory + "/FPR_Rejected/" + std::to_string(lc_prev_idx) + "_" + std::to_string(lc_curr_idx) + \
                                           "_" + std::to_string(overlap_percen*100) + "%_cloud_cur_guess.pcd", *currKeyframeCloud_tmp); // scan
                pcl::io::savePCDFileBinary(pgo_scan_directory + "/FPR_Rejected/" + std::to_string(lc_prev_idx) + "_" + std::to_string(lc_curr_idx) + \
                                           "cloud_target.pcd", *targetKeyframeCloud_tmp); // scan
            } catch (pcl::PCLException e)
            {
                ROS_ERROR("%s", e.what());
            }
        }
    }
    else
    {
//        auto complete_graph = isam_->getFactorsUnsafe();
//        opt_debug_file << "aft opt: " << std::endl;
//        printGraph(complete_graph);
        opt_debug_file <<  "[FPR]: accept, everything is ok." << std::endl;
        loop_corr_counter++;
        pubPath();

        factors_toremove.clear();
        gts_graph_.resize(0);
        gts_graph_recover_.resize(0);
        gts_init_vals_.clear();

        VOXEL_LOC position; position.x = (int)(lc_prev_idx),position.y = (int)(lc_curr_idx),position.z = (int)(0);
        auto icp_tranform = GTSPoseToPose6D(lc_pose_uomap[position]);
        position.x = (int)(last_lc_prev_idx),position.y = (int)(last_lc_curr_idx),position.z = (int)(0);
        auto icp_tranform_last = GTSPoseToPose6D(lc_pose_uomap[position]);
        lc_file << last_lc_prev_idx << " " << last_lc_curr_idx << "  " << icp_tranform_last.x << "  " << icp_tranform_last.y << "  " << icp_tranform_last.z <<
              "  " << icp_tranform_last.roll << "  " << icp_tranform_last.pitch << "  " << icp_tranform_last.yaw << std::endl;
        lc_file << lc_prev_idx      << " " << lc_curr_idx      << "  " << icp_tranform.x      << "  " << icp_tranform.y      << "  " << icp_tranform.z      <<
              "  " << icp_tranform.roll      << "  " << icp_tranform.pitch      << "  " << icp_tranform.yaw      << std::endl;

if (multisession_mode == 1)
{
        if (relocal_status == 0)   // set relocalizaion status as done (= 1) in multisession mode
            relocal_status = 1;
}
        if (loop_corr_counter == 1)
            pubLoopClosureVisualization(last_lc_prev_idx, last_lc_curr_idx);

        pubLoopClosureVisualization(lc_prev_idx, lc_curr_idx);
        correctPosesAndSaveTxt();
#ifdef marg
        if (lc_prev_idx >= 0)
            marginalizeKPPairFactor(lc_prev_idx, lc_curr_idx, false);
if (multisession_mode == 1)
{
        opt_debug_file <<  "last_lc_prev_idx " << last_lc_prev_idx << ", lc_prev_idx " << lc_prev_idx << std::endl;
        opt_debug_file <<  "last_lc_curr_idx " << last_lc_curr_idx << ", lc_curr_idx " << lc_curr_idx << std::endl;

        if (relocal_status == 1 && last_lc_prev_idx < 0 && lc_prev_idx < 0)  // if relocalization is done, and LCs are against prior map, in multisession mode
            marginalizeKPPairFactor(last_lc_curr_idx, lc_curr_idx, false);
}
#endif

        passed_dis_atlast_lc = passed_dis;
        if(dosave_pcd)
        {
            try
            {
                pcl::io::savePCDFileBinary(pgo_scan_directory + "/FPR_Accepted/" + std::to_string(lc_prev_idx) + "_" + std::to_string(lc_curr_idx) + \
                                           "_" + std::to_string(overlap_percen*100) + "%_cloud_cur_guess.pcd", *currKeyframeCloud_tmp); // scan
                pcl::io::savePCDFileBinary(pgo_scan_directory + "/FPR_Accepted/" + std::to_string(lc_prev_idx) + "_" + std::to_string(lc_curr_idx) + \
                                           "cloud_target.pcd", *targetKeyframeCloud_tmp); // scan
            } catch (pcl::PCLException e)
            {
                ROS_ERROR("%s", e.what());
            }
        }
    }

    last_lc_prev_idx = lc_prev_idx;
    last_lc_curr_idx = lc_curr_idx;
    just_loop_closure = false;
}

void insertKPPairConstraint(const pcl::PointCloud<PointType>::Ptr &cloud_tmp, const int prev_idx, const int curr_idx, const int option)
{
    pcl::PointCloud<PointType>::Ptr cloud_prev(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr cloud_curr(new pcl::PointCloud<PointType>());
    if (option == 1)
    {
        gtsam::Matrix3 R_prev, R_curr;
        gtsam::Vector3 t_prev, t_curr;

        if (keyframes_[prev_idx].KeyTime.toNSec() > jump_time)
            Pose6DToEigenRT(keyframes_[prev_idx].KeyPose, R_prev, t_prev);
        else
        {
          gtsam::Pose3 pose_prev_bjump = Pose6DToGTSPose(keyframes_[prev_idx].KeyPose);
          opt_debug_file <<  "[Jump]: prev pose node(" << prev_idx << ") remap from" << pose_prev_bjump.translation();
          gtsam::Pose3 pose_prev = subpose_afterjump*subpose_beforejump.inverse()*pose_prev_bjump;
          opt_debug_file <<  " -> " << pose_prev.translation() << std::endl;
          Pose6DToEigenRT(GTSPoseToPose6D(pose_prev), R_prev, t_prev);
        }
        if (keyframes_[curr_idx].KeyTime.toNSec() > jump_time)
          Pose6DToEigenRT(keyframes_[curr_idx].KeyPose, R_curr, t_curr);
        else
        {
          gtsam::Pose3 pose_curr_bjump = Pose6DToGTSPose(keyframes_[curr_idx].KeyPose);
          opt_debug_file <<  "[Jump]: cur pose node(" << curr_idx << ") remap from" << pose_curr_bjump.translation();
          gtsam::Pose3 pose_curr = subpose_afterjump*subpose_beforejump.inverse()*pose_curr_bjump;
          opt_debug_file <<  " -> " << pose_curr.translation() << std::endl;
          Pose6DToEigenRT(GTSPoseToPose6D(pose_curr), R_curr, t_curr);
        }

        gtsam::Matrix3 R_prev_inv = R_prev.transpose();
        t_prev = -R_prev_inv*t_prev;
        gtsam::Matrix3 R_curr_inv = R_curr.transpose();
        t_curr = -R_curr_inv*t_curr;

        std::vector<std::tuple<gtsam::Vector3, gtsam::Vector3, float>> corner_pairs;
        int half = cloud_tmp->size()/2;
        for (int i = 0; i < half; i++)
        {
            gtsam::Vector3 pt_curr(cloud_tmp->points[i].x, cloud_tmp->points[i].y, cloud_tmp->points[i].z);
            gtsam::Vector3 pt_prev(cloud_tmp->points[i+half].x, cloud_tmp->points[i+half].y, cloud_tmp->points[i+half].z);

            gtsam::Point3 pt_prev_body = R_prev_inv*pt_prev + t_prev;
            gtsam::Point3 pt_curr_body = R_curr_inv*pt_curr + t_curr;

            float dis = pt_prev_body.norm() + pt_curr_body.norm();
            corner_pairs.push_back({pt_curr_body, pt_prev_body, dis}) ;
        }

        std::sort(corner_pairs.begin(), corner_pairs.end(), keyPointDisSort);
        int max_num = corner_pairs.size() > pairfactor_num ? pairfactor_num:corner_pairs.size();

        for (int i = 0; i < max_num; i++)
        {
            gtsam::Point3 pt_prev_body = std::get<1>(corner_pairs[i]);
            gtsam::Point3 pt_curr_body = std::get<0>(corner_pairs[i]);

            boost::shared_ptr<gtsam::TunningPointPairsFactor<gtsam::Pose3>> tmp
                (new gtsam::TunningPointPairsFactor<gtsam::Pose3>(prev_idx, curr_idx, pt_prev_body, pt_curr_body, point_noise_));

            gts_graph_.add(tmp);
            gts_graph_recover_.add(tmp);

            PointType a_pt;
            a_pt.x = pt_prev_body[0]; a_pt.y = pt_prev_body[1]; a_pt.z = pt_prev_body[2];
            cloud_prev->push_back(a_pt);
            a_pt.x = pt_curr_body[0]; a_pt.y = pt_curr_body[1]; a_pt.z = pt_curr_body[2];
            cloud_curr->push_back(a_pt);
        }
        opt_debug_file <<  "[Add Adj KPF]: " << prev_idx << " " << curr_idx << " " << cloud_curr->size() << " " << cloud_prev->size()  << " " << cloud_tmp->size()/2 << std::endl;
        std::cout      <<  "[Add Adj KPF]: " << prev_idx << " " << curr_idx << std::endl;
    }
    else if (option == 2)
    {
        pcl::PointCloud<PointType>::Ptr tmp_prev (new pcl::PointCloud<PointType>);
        pcl::PointCloud<PointType>::Ptr tmp_curr (new pcl::PointCloud<PointType>);

        int half = cloud_tmp->size()/2;
        for (int i = 0; i < half; i+=3)
        {
            tmp_curr->push_back(cloud_tmp->points[i]);
            tmp_curr->push_back(cloud_tmp->points[i+1]);
            tmp_curr->push_back(cloud_tmp->points[i+2]);
        }

        for (int i = half; i < cloud_tmp->size(); i+=3)
        {
            tmp_prev->push_back(cloud_tmp->points[i]);
            tmp_prev->push_back(cloud_tmp->points[i+1]);
            tmp_prev->push_back(cloud_tmp->points[i+2]);
        }
        gtsam::Pose3 T_prev;
if (multisession_mode == 1)
{
        if(prev_idx < 0)
            T_prev = Pose6DToGTSPose(keyframes_prior_[wrap4Containner(prev_idx)].KeyPose).inverse();
        else
            T_prev = Pose6DToGTSPose(keyframes_[prev_idx].KeyPose).inverse();
}
else
        T_prev = Pose6DToGTSPose(keyframes_[prev_idx].KeyPose).inverse();

        gtsam::Pose3 T_curr = Pose6DToGTSPose(keyframes_[curr_idx].KeyPose).inverse();
        pcl::transformPointCloud(*tmp_prev, *cloud_prev, GTSPoseToEigenM4f(T_prev));
        pcl::transformPointCloud(*tmp_curr, *cloud_curr, GTSPoseToEigenM4f(T_curr));

        for (int i = 0; i < cloud_curr->size(); i++)
        {
            gtsam::Point3 pt_prev_body(cloud_prev->points[i].x, cloud_prev->points[i].y, cloud_prev->points[i].z);
            gtsam::Point3 pt_curr_body(cloud_curr->points[i].x, cloud_curr->points[i].y, cloud_curr->points[i].z);

            boost::shared_ptr<gtsam::TunningPointPairsFactor<gtsam::Pose3>> tmp
                (new gtsam::TunningPointPairsFactor<gtsam::Pose3>(wrap4Gtsam(prev_idx), curr_idx, pt_prev_body, pt_curr_body, lc_point_noise_));

            gts_graph_.add(tmp);
        }

        opt_debug_file <<  "[Add LC KPF]: " << prev_idx << " " << curr_idx << " " << cloud_curr->size() << " " << cloud_prev->size()  << " " << cloud_tmp->size()/2 << std::endl;
        std::cout      <<  "[Add LC KPF]: " << prev_idx << " " << curr_idx << std::endl;
    }
    VOXEL_LOC prev_curr_idx;
    prev_curr_idx.x = (int)(prev_idx),prev_curr_idx.y = (int)(curr_idx),prev_curr_idx.z = (int)(0);
    auto iter = cornerpairs_uomap.find(prev_curr_idx);
    if(iter == cornerpairs_uomap.end())           // store KP pairs info into containner cornerpairs_uomap
        cornerpairs_uomap[prev_curr_idx] = {cloud_prev, cloud_curr};
}

// odom constraints
bool addOdomFactor()
{
    if (graphpose_count == keyframes_.size()) return false;

    if (!has_add_prior_node)
    {
        gtsam::Pose3 pose_origin = Pose6DToGTSPose(keyframes_[0].KeyPose);
if (multisession_mode == 1)
        addInitialPoseValues(0, pose_origin);
else
        addPriorPoses(0, pose_origin);

        graphpose_count++;
        has_add_prior_node = true;
        poses_raw_file << 0 << " " << pose_origin.x() << " " << pose_origin.y() << " " << pose_origin.z() << " " << std::endl;
        x_range_max = pose_origin.x(); y_range_max = pose_origin.y(); z_range_max = pose_origin.z();
        x_range_min = pose_origin.x(); y_range_min = pose_origin.y(); z_range_min = pose_origin.z();
    }

    while (graphpose_count < keyframes_.size())
    {
        const int prev_node_idx = graphpose_count - 1;
        const int curr_node_idx = graphpose_count;
        std::cout <<       "[Add Odom Factor]: "  << prev_node_idx << " " << curr_node_idx << " " << std::endl;
        opt_debug_file <<  "[Add Odom Factor]: "  << prev_node_idx << " " << curr_node_idx << " " << std::endl;
        gtsam::Pose3 pose_prev, pose_curr;

        if (keyframes_[prev_node_idx].KeyTime.toNSec() > jump_time)
            pose_prev = Pose6DToGTSPose(keyframes_[prev_node_idx].KeyPose);
        else
        {
            gtsam::Pose3 pose_prev_bjump = Pose6DToGTSPose(keyframes_[prev_node_idx].KeyPose);
            opt_debug_file <<  "[Add Odom Factor]: due to pose jump, prev pose node(" << prev_node_idx << ") remap from" << pose_prev_bjump.translation();
            pose_prev = subpose_afterjump*subpose_beforejump.inverse()*pose_prev_bjump;
            opt_debug_file <<  " -> " << pose_prev.translation() << std::endl;
        }

        if (keyframes_[curr_node_idx].KeyTime.toNSec() > jump_time)
            pose_curr = Pose6DToGTSPose(keyframes_[curr_node_idx].KeyPose);
        else
        {
            gtsam::Pose3 pose_curr_bjump = Pose6DToGTSPose(keyframes_[curr_node_idx].KeyPose);
            opt_debug_file <<  "[Add Odom Factor]: due to pose jump, curr pose node(" << curr_node_idx << ") remap from" << pose_curr_bjump.translation();
            pose_curr =  subpose_afterjump*subpose_beforejump.inverse()*pose_curr_bjump;
            opt_debug_file <<  " -> " << pose_curr.translation() << std::endl;
        }

//        opt_debug_file <<  " pose_prev: " << pose_prev.translation() << " pose_cur: " << pose_curr.translation() << std::endl;
        gts_graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(prev_node_idx, curr_node_idx, pose_prev.between(pose_curr), pose_noise_));
        gts_graph_recover_.add(gtsam::BetweenFactor<gtsam::Pose3>(prev_node_idx, curr_node_idx, pose_prev.between(pose_curr), pose_noise_));
        addInitialPoseValues(curr_node_idx, pose_curr);
        graphpose_count++;
        poses_raw_file << curr_node_idx << " " << pose_curr.x() << " " << pose_curr.y() << " " << pose_curr.z() << " " << std::endl;
        x_range_max = pose_curr.x()>x_range_max?pose_curr.x():x_range_max;
        y_range_max = pose_curr.y()>y_range_max?pose_curr.y():y_range_max;
        z_range_max = pose_curr.z()>z_range_max?pose_curr.z():z_range_max;
        x_range_min = pose_curr.x()<x_range_min?pose_curr.x():x_range_min;
        y_range_min = pose_curr.y()<y_range_min?pose_curr.y():y_range_min;
        z_range_min = pose_curr.z()<z_range_min?pose_curr.z():z_range_min;

//        if (curr_node_idx == 487)
//        {
//          lc_prev_idx = 0;
//          lc_curr_idx = 487;
//          Eigen::Matrix4f icp_trans;
////          icp_trans << 0.999734103680, 0.022584797814, -0.004629123956, 0.043591294438,
////-0.022666096687, 0.999574959278, -0.018337761983, 0.424079865217,
////0.004213014618, 0.018437810242, 0.999821186066, -2.445192813873,
////0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000;
//          icp_trans << 0.999601721764, -0.028179764748, 0.001382074784, -0.054362889379,
//0.028079377487, 0.998424112797, 0.048585116863, -0.360815644264,
//-0.002749013249, -0.048526994884, 0.998818039894, 2.503658294678,
//0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000;
//          Eigen::Matrix3f R = icp_trans.block<3,3>(0,0);
//          Eigen::Vector3f t = icp_trans.block<3,1>(0,3);
//          Eigen::Quaternionf q(R);
//          double roll, pitch, yaw;
//          tf::Matrix3x3(tf::Quaternion(q.x(), q.y(), q.z(), q.w())).getRPY(roll, pitch, yaw);
//          gtsam::Pose3 oh_pose_oc = gtsam::Pose3(gtsam::Rot3::RzRyRx(roll, pitch, yaw), gtsam::Point3(t.x(), t.y(), t.z()));
//          gtsam::Pose3 oc_pose_bc = Pose6DToGTSPose(keyframes_[lc_curr_idx].KeyPose);
//          gtsam::Pose3 oh_pose_bh = Pose6DToGTSPose(keyframes_[lc_prev_idx].KeyPose);
//          gtsam::Pose3 bh_pose_oh = oh_pose_bh.inverse();
//          gtsam::Pose3 bh_pose_bc = bh_pose_oh*oh_pose_oc*oc_pose_bc;

//          opt_debug_file << "Manual insertion" << std::endl;
//          gtsam::Vector noise_vec6(6);
//          noise_vec6 << 1e-6, 1e-6, 1e-6, 1e-3, 1e-3, 1e-3;
//          loopclousure_noise_ = gtsam::noiseModel::Diagonal::Variances(noise_vec6);  //uncomment this line for no robust estimator
//          //loopclousure_noise_ = gtsam::noiseModel::Robust::Create(
//          //    gtsam::noiseModel::mEstimator::Cauchy::Create(1),
//          //    gtsam::noiseModel::Diagonal::Variances(noise_vec6));

//          gts_graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(lc_prev_idx, lc_curr_idx, bh_pose_bc, loopclousure_noise_));
//          just_loop_closure = true;
//          optimize();
//        }

    }
    return true;
}

void main_pgo()
{
    while(1)
    {
        sleep(0.001);
        if (odom_buf_.empty() || cloud_buf_.empty())
            continue;
        mtx_sub_.lock();
        {
            // turn raw messages into stuff in a containner
            while (!odom_buf_.empty() && odom_buf_.front()->header.stamp.toSec() < cloud_buf_.front()->header.stamp.toSec())
                odom_buf_.pop_front();
            if (odom_buf_.empty())
            {
                mtx_sub_.unlock();
                continue;
            }
            if (fabs(odom_buf_.front()->header.stamp.toSec() - cloud_buf_.front()->header.stamp.toSec()) > 1)
                ROS_WARN_STREAM("Too large odom lidar msg time stamp difference!");

            KeyFrame akeyframe;
            auto cloud_buf_front = cloud_buf_.front();
            auto odom_buf_front = odom_buf_.front();
            pcl::PointCloud<PointType>::Ptr cloud_tmp(new pcl::PointCloud<PointType>());
            pcl::fromROSMsg(*(cloud_buf_front), *cloud_tmp);
            akeyframe.KeyCloud = cloud_tmp;

            akeyframe.KeyPose = OdomMsgToPose6D(odom_buf_front);
            if (!keyframes_.empty())
                passed_dis += sqrt((akeyframe.KeyPose.x - keyframes_.back().KeyPose.x)*(akeyframe.KeyPose.x - keyframes_.back().KeyPose.x) + \
                                   (akeyframe.KeyPose.y - keyframes_.back().KeyPose.y)*(akeyframe.KeyPose.y - keyframes_.back().KeyPose.y) + \
                                   (akeyframe.KeyPose.z - keyframes_.back().KeyPose.z)*(akeyframe.KeyPose.z - keyframes_.back().KeyPose.z));
            akeyframe.KeyTime = cloud_buf_front->header.stamp;
            keyframes_.push_back(akeyframe);
            cloud_buf_.pop_front();
            odom_buf_.pop_front();
            opt_debug_file << "[Add Subamp Info]: akeyframe.KeyCloud: " << akeyframe.KeyCloud->size() << std::endl;
            opt_debug_file << "[Add Subamp Info]: akeyframe.KeyTime: " << akeyframe.KeyTime.toNSec() << std::endl;
            opt_debug_file << "[Add Subamp Info]: akeyframe.KeyPose: " << akeyframe.KeyPose.x << " " << akeyframe.KeyPose.y << " " << akeyframe.KeyPose.z << std::endl;
            opt_debug_file << "[Add Subamp Info]: passed_dis: " <<passed_dis;
            opt_debug_file << ", keyframes size(): " << keyframes_.size() << std::endl;
        }
        mtx_sub_.unlock();

        mtx_pgo_.lock();
        auto start = std::chrono::system_clock::now();
        if (addOdomFactor())   // when a new key frame appears (addOdomFactor == true), do some thing
        {
            while (!loopclosure_buf_.empty() && !just_loop_closure)
            {
                auto &lc_msg = loopclosure_buf_.front();            // loopclosure_buf_ contain a LC info: index pair & icp tranformation
                const int prev_node_idx = lc_msg->covariance[1];
                const int curr_node_idx = lc_msg->covariance[0];
                opt_debug_file << "Detected LC: " << prev_node_idx << " " << curr_node_idx << std::endl;
                if (curr_node_idx + 1 > keyframes_.size())
                {
//                    ROS_WARN_STREAM("Loop clousre factor ahead odom factor!");
                    opt_debug_file <<  "[Add LC Factor]: Loop clousre factor ahead odom factor!" << std::endl;
                    break;
                }

                lc_curr_idx = curr_node_idx;
                lc_prev_idx = prev_node_idx;
                gtsam::Pose3 lc_transform = GeoPoseMsgToGTSPose(lc_msg->pose);
                gtsam::Pose3 oh_pose_oc = lc_transform;
                gtsam::Pose3 oh_pose_bh;
if (multisession_mode == 1)
{
                if (prev_node_idx < 0)                    // LC against prior map
                {
                    if (-prev_node_idx-1 >= keyframes_prior_.size())
                    {
                        opt_debug_file << "[Warnning]: fetch index large than prior keyframe number" << std::endl;
                        break;
                    }
                    oh_pose_bh = Pose6DToGTSPose(keyframes_prior_[-prev_node_idx-1].KeyPose);
                }
                else
                    oh_pose_bh = Pose6DToGTSPose(keyframes_[prev_node_idx].KeyPose);
}
else
                oh_pose_bh = Pose6DToGTSPose(keyframes_[prev_node_idx].KeyPose);

                gtsam::Pose3 oc_pose_bc = Pose6DToGTSPose(keyframes_[curr_node_idx].KeyPose);
                gtsam::Pose3 bh_pose_oh = oh_pose_bh.inverse();
                gtsam::Pose3 bh_pose_bc = bh_pose_oh*oh_pose_oc*oc_pose_bc;
                gtsam::Pose3 pose_odom = oh_pose_bh.between(oc_pose_bc);
                Eigen::Matrix4f guess = GTSPoseToEigenM4f(bh_pose_bc);
                float overlap_score;
if (multisession_mode == 1)
{
                if (prev_node_idx < 0)                    // LC against prior map
                    overlap_score = calculateOverlapScore(keyframes_[curr_node_idx].KeyCloud, keyframes_prior_[-prev_node_idx-1].KeyCloud,
                                                          GTSPoseToEigenM4f(oc_pose_bc), GTSPoseToEigenM4f(oh_pose_bh),
                                                          guess, vs_for_ovlap, 150000, prev_node_idx, curr_node_idx);
                else
                    overlap_score = calculateOverlapScore(keyframes_[curr_node_idx].KeyCloud, keyframes_[prev_node_idx].KeyCloud,
                                                          GTSPoseToEigenM4f(oc_pose_bc), GTSPoseToEigenM4f(oh_pose_bh),
                                                          guess, vs_for_ovlap, 150000, prev_node_idx, curr_node_idx);
}else
                overlap_score = calculateOverlapScore(keyframes_[curr_node_idx].KeyCloud, keyframes_[prev_node_idx].KeyCloud,
                                                      GTSPoseToEigenM4f(oc_pose_bc), GTSPoseToEigenM4f(oh_pose_bh),
                                                      guess, vs_for_ovlap, 150000, prev_node_idx, curr_node_idx);

                overlap_percen = overlap_score;
                if (overlap_score != 0)
                {
                    opt_debug_file <<  "[FPR]: overlap ratio is " << overlap_score << std::endl;
                    std::cout      <<  "[FPR]: overlap ratio is " << overlap_score << std::endl;
                }
                loopclosure_buf_.pop_front();

#ifndef robust_estimator
                if (overlap_score < overlap_score_thr)             // check LC cloud pair overlap ratio
                {
                    opt_debug_file <<  "[FPR]: reject, too small overlap." << std::endl;
                    removeLC(curr_node_idx, prev_node_idx);
                    continue;
                }

                float x_range = x_range_max - x_range_min; float y_range = y_range_max - y_range_min; float z_range = z_range_max - z_range_min;
                if (lc_pose_uomap.empty()) // hard for the first LC to pass
                {
                    opt_debug_file <<  "[FPR]: need to wait for one more loop closure." << std::endl;
                    wait_1more_loopclosure = true;
                }
                // hard for long distance LC to pass
                else if (lc_pose_uomap.size() > 2 && !wait_1more_loopclosure && lc_transform.translation().norm() > 0.1*std::max(std::max(x_range, y_range),z_range) )
                {
                    opt_debug_file <<  "[FPR]: long key point pair distance (" << lc_transform.translation().norm() << "m), need to wait for one more loop closure." << std::endl;
                    opt_debug_file <<  "x_range: " << x_range << " y_range: " << y_range << " z_range: " << z_range << std::endl;
                    if (lc_transform.translation().norm() > 0.5*std::max(std::max(x_range, y_range),z_range))   // TODO: how to verify unexpected large jump LC
                    {
                        removeLC(curr_node_idx, prev_node_idx);
                        continue;
                    }
                    wait_1more_loopclosure = true;
                }
                else
                {
                    wait_1more_loopclosure = false;
                }
#endif

                VOXEL_LOC position; position.x = (int)(prev_node_idx),position.y = (int)(curr_node_idx),position.z = (int)(0);
                if (lc_pose_uomap.find(position) == lc_pose_uomap.end())      // avoid unexpected reduantant LC
                {
                    lc_pose_uomap[position] = bh_pose_bc;
                    //opt_debug_file << "lc_pose_uomap inserts pose for (" << prev_node_idx << "," << curr_node_idx << ")" << std::endl;
#ifndef robust_estimator
                    if (wait_1more_loopclosure)
                    {
                        continue;
                    }
#endif
                }

#ifdef robust_estimator
                gtsam::Vector noise_vec6(6);
                noise_vec6 << 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1;
//                loopclousure_noise_ = gtsam::noiseModel::Diagonal::Variances(noise_vec6);  //uncomment this line for no robust estimator
                loopclousure_noise_ = gtsam::noiseModel::Robust::Create(
                    gtsam::noiseModel::mEstimator::Cauchy::Create(1),
                    gtsam::noiseModel::Diagonal::Variances(noise_vec6));
                gts_graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(prev_node_idx, curr_node_idx, bh_pose_bc, loopclousure_noise_));
                opt_debug_file <<  "[Add LC Factor]: " << prev_node_idx << " " << curr_node_idx << std::endl;
                std::cout      <<  "[Add LC Factor]: " << prev_node_idx << " " << curr_node_idx << std::endl;
#endif

                // loop closure tri constraint
                int lc_count = 0;
                opt_debug_file << "LC KPF queue size is " << LCT_buf_.size() << " now." << std::endl;
                std::deque<sensor_msgs::PointCloud2ConstPtr> LCT_buf_tmp;
                while (!LCT_buf_.empty())   // iterate over LC KP messgage buffer
                {
                    auto LCTmsg = LCT_buf_.front();
                    pcl::PointCloud<PointType>::Ptr cloud_tmp(new pcl::PointCloud<PointType>());
                    pcl::fromROSMsg(*(LCTmsg), *cloud_tmp);
                    int prev_idx_lct = cloud_tmp->points[0].intensity;
                    int curr_idx_lct = cloud_tmp->points[1].intensity;

                    opt_debug_file <<  "LC KPF queue front: " << prev_idx_lct << " " << curr_idx_lct ;
                    if (curr_idx_lct + 1 > keyframes_.size())
                    {
//                        ROS_WARN_STREAM("Loop clousre KP factor ahead odom factor!");
                        opt_debug_file <<  "Loop clousre KP factor ahead odom factor!" << std::endl;
                        break;
                    }
                    LCT_buf_.pop_front();
                    VOXEL_LOC position2; position2.x = (int)(prev_idx_lct),position2.y = (int)(curr_idx_lct),position2.z = (int)(0);
                    if (lc_pose_uomap.find(position2) != lc_pose_uomap.end())    // if a detected LC pass senatation check, e.g., overlap check,
                    {                                                            // you can find it from the containner
                        just_loop_closure = true;
                        lc_count++;
if (multisession_mode == 1)
{
                        if (prev_idx_lct >= 0)    // LC against live map
                            insertKPPairConstraint(cloud_tmp,  prev_idx_lct, curr_idx_lct, 2);
                        if (prev_idx_lct < 0)     // LC against prior map
                        {
                            if(lc_count == 2 && relocal_status == -1) // if first two LC found & not yet try relocalization
                            {
                                addPriorPosesFromPriorInfo(prev_idx_lct, curr_idx_lct);
                                addPriorPosesFromPriorInfo(last_lc_prev_idx, last_lc_curr_idx);
                                relocal_status = 0;  //relocal_status = 0 means we not sure relocalization successful or not
                                break;
                            }

                            if (lc_count == 1 && relocal_status == -1)  // if first one LC found & not yet try relocalization
                            {
                                last_lc_prev_idx = prev_idx_lct;
                                last_lc_curr_idx = curr_idx_lct;
                            }
                        }
                        if (lc_count == 1 && relocal_status == 1) // if relocalization already done, just need one LC for opt.
                        {
                            if (prev_idx_lct < 0)
                                addPriorPosesFromPriorInfo(prev_idx_lct, curr_idx_lct);   // add LC against prior map as prior factor
                            break;
                        }
}
else
{
#ifndef robust_estimator
                        insertKPPairConstraint(cloud_tmp,  prev_idx_lct, curr_idx_lct, 2);
#endif
}
                    }
                    else
                    {
                        LCT_buf_tmp.push_back(LCTmsg);
                        opt_debug_file <<  "Skip." << std::endl;
                    }
                }
                LCT_buf_ = LCT_buf_tmp;
                opt_debug_file << "LC KPF queue size is " << LCT_buf_.size() << " now." << std::endl;
            }

            while(!interT_buf_.empty())
            {
                auto interT_front = interT_buf_.front();
                pcl::PointCloud<PointType>::Ptr cloud_tmp(new pcl::PointCloud<PointType>());
                pcl::fromROSMsg(*(interT_front), *cloud_tmp);
                const int prev_node_idx = cloud_tmp->points[0].intensity;
                const int curr_node_idx = cloud_tmp->points[1].intensity;

                if (cloud_tmp->points[0].x == 0 && cloud_tmp->points[0].y == 0 && cloud_tmp->points[0].z == 0 &&
                    cloud_tmp->points[1].x == 0 && cloud_tmp->points[1].y == 0 && cloud_tmp->points[1].z == 0)    // invalid Adj. KP factor message
                {
                    interT_buf_.pop_front();
                    opt_debug_file <<  "[Add Adjacent KPF]: adjacent KP factor absent! False Positive Rejection might not work due to this." << std::endl;
//                    ROS_WARN_STREAM("Adjacent KP factor absent! False Positive Rejection might not work due to this.");
                    continue;
                }

                if (curr_node_idx + 1 > keyframes_.size())
                {
                    opt_debug_file <<  "[Add Adjacent KPF]: loop clousre factor ahead odom factor!" << std::endl;
//                    ROS_WARN_STREAM("Adjacent KP factor ahead odom factor!");
                    break;
                }
#ifndef robust_estimator
                insertKPPairConstraint(cloud_tmp, prev_node_idx, curr_node_idx, 1);
#endif
                interT_buf_.pop_front();
            }
//            while(!inter2T_buf_.empty())
//            {
//                auto inter2T_front = inter2T_buf_.front();
//                pcl::PointCloud<PointType>::Ptr cloud_tmp(new pcl::PointCloud<PointType>());
//                pcl::fromROSMsg(*(inter2T_front), *cloud_tmp);
//                const int prev_node_idx = cloud_tmp->points[0].intensity;
//                const int curr_node_idx = cloud_tmp->points[1].intensity;

//                if (cloud_tmp->points[0].x == 0 && cloud_tmp->points[0].y == 0 && cloud_tmp->points[0].z == 0 &&
//                    cloud_tmp->points[1].x == 0 && cloud_tmp->points[1].y == 0 && cloud_tmp->points[1].z == 0)
//                {
//                    inter2T_buf_.pop_front();
//                    continue;
//                }

//                if (curr_node_idx + 1 > keyframes_.size()){
////                    ROS_WARN_STREAM("Adjacent KP 2 factor ahead odom factor!");
//                    opt_debug_file <<  "[Add Adjacent KPF]: loop clousre factor ahead odom factor!" << std::endl;
//                    break;
//                }
//    #ifndef robust_estimator
//                insertKPPairConstraint(cloud_tmp, prev_node_idx, curr_node_idx, 1);
//    #endif
//                inter2T_buf_.pop_front();
//            }

if (multisession_mode == 1)
{
            if (relocal_status > -1)   // only optimize when relocalization is done (=1) or waiting to verify (=0)
                optimize();
}
else
{
            optimize();
}
            opt_debug_file <<  "------------------------------" << std::endl;
        }
        auto end = std::chrono::system_clock::now();
        std::chrono::duration<double, std::milli> elapsed_ms = std::chrono::duration<double,std::milli>(end - start);
        time_full_pgo_thread << elapsed_ms.count() << std::endl;
        mtx_pgo_.unlock();
    }
}

pcl::PointCloud<PointType> fullMapToSave;
int loop_corr_counter_last = 0;
void pubAndSaveGloablMap(bool savemap = true)
{
    if (keyframes_.empty())  return;
    if (loop_corr_counter == loop_corr_counter_last) return;
    if (!fullMapToSave.empty())      pcl::PointCloud<PointType> ().swap(fullMapToSave);
    mtx_keyf_read_.lock();
    auto keyframes = keyframes_;
    mtx_keyf_read_.unlock();
    for (int node_idx=0; node_idx < keyframes.size(); node_idx++)
    {
        if (!keyframes_[node_idx].pose_opt_set) continue;
        gtsam::Pose3 T1 = Pose6DToGTSPose(keyframes[node_idx].KeyPose).inverse();
        gtsam::Pose3 T2 = Pose6DToGTSPose(keyframes[node_idx].KeyPoseOpt).inverse();
        gtsam::Pose3 _2T1_ = T2.between(T1);

        if (keyframes[node_idx].KeyCloud->empty())    continue;
        //DownsampleCloud(keyframes_[node_idx].KeyCloud, 0.2);
        //pcl::PointCloud<PointType>::Ptr tmpToSave (new pcl::PointCloud<PointType>);
        //pcl::transformPointCloud(*(keyframes_[node_idx].KeyCloud), *tmpToSave, GTSPoseToEigenM4f(_2T1_));
        auto Rt = GTSPoseToEigenM4f(_2T1_);
        Eigen::Matrix3f R = Rt.block<3,3>(0,0);
        Eigen::Vector3f t = Eigen::Vector3f(Rt(0,3), Rt(1,3), Rt(2,3));
        int pt_skip = max(1, int(pcdsave_step));
        for (size_t pidx = 0; pidx < keyframes[node_idx].KeyCloud->points.size(); pidx += pt_skip)
        {
            auto a_pt = keyframes_[node_idx].KeyCloud->points[pidx];
            Eigen::Vector3f xyz_cor = R * Eigen::Vector3f(a_pt.x, a_pt.y, a_pt.z) + t;
            PointType a_pt_cor; a_pt_cor.x = xyz_cor(0); a_pt_cor.y = xyz_cor(1); a_pt_cor.z = xyz_cor(2);
            fullMapToSave.push_back(a_pt_cor);
        }
        std::cout      << "Correcting submap " << node_idx << ", total map size: " << fullMapToSave.size() << std::endl;
        opt_debug_file << "Correcting submap " << node_idx << ", total map size: " << fullMapToSave.size() << std::endl;
    }
    if (fullMapToSave.empty()) return;
    //DownsampleCloud(fullMapToSave, 0.5); // !!!be careful about resolution, too small will cause memory bug or ros message error.
    sensor_msgs::PointCloud2 laserCloudMapPGOMsg;
    pcl::toROSMsg(fullMapToSave, laserCloudMapPGOMsg);
    laserCloudMapPGOMsg.header.frame_id = "/camera_init";
    if (pubAndSaveGloablMapAftPGO.getNumSubscribers() != 0)
        pubAndSaveGloablMapAftPGO.publish(laserCloudMapPGOMsg);
    ros::spinOnce();
    loop_corr_counter_last = loop_corr_counter;

//#ifdef save_pcd
    if (!fullMapToSave.empty() && savemap)
    {
        opt_debug_file << "[SaveMap]: try saving full corrected map." << std::endl;
        pcl::io::savePCDFileBinary(save_directory + "cloud_result.pcd", fullMapToSave);
        opt_debug_file << "[SaveMap]: result map saved as cloud_result.pcd with size: " << fullMapToSave.size() << std::endl;
        std::cout << "[SaveMap]: result map saved as " << save_directory << "cloud_result.pcd with size: " << fullMapToSave.size() << std::endl;
    }
//#endif

}

void processRvizMap(void)
{
    if (!dopub_corrmap) return;
    while(1)
    {
        sleep(1);
        pubAndSaveGloablMap(false);
    }
}

pcl::PointCloud<PointType>::Ptr fullCorrected_p(new pcl::PointCloud<PointType>);
pcl::PointCloud<PointType>::Ptr tmp_p (new pcl::PointCloud<PointType>);
pcl::PointCloud<PointType>::Ptr cloud_ds_p(new pcl::PointCloud<PointType>);
void savePrior()                                     // Multisession mode function, for storing first session's result
{
    std::cout      << "[SavePrior]: save_one_time "  << std::endl;
    opt_debug_file << "[SavePrior]: save_one_time "  << std::endl;
    fullCorrected_p->clear();
    fullCorrected_p->points.reserve(200000000);
    tmp_p->clear();
    cloud_ds_p->clear();
    pcl::VoxelGrid<PointType> sor;
    float leafsize = 0.5;
    sor.setLeafSize(leafsize, leafsize, leafsize);
    for (int i = 0; i < keyframes_.size(); i++)
    {
        auto &akeyframe = keyframes_[i];
        if (!akeyframe.pose_opt_set)
            continue;

        int cloud_size = akeyframe.KeyCloud->size();
        if (cloud_size==0) continue;

        try
        {
            *fullCorrected_p += *(akeyframe.KeyCloud);
        }catch(std::bad_alloc)
        {
            std::cerr << "std::bad_alloc" << std::endl;
            continue;
        }

        if (akeyframe.pose_opt_set)
        {
            gtsam::Pose3 T1 = Pose6DToGTSPose(akeyframe.KeyPose).inverse();
            gtsam::Pose3 T2 = Pose6DToGTSPose(akeyframe.KeyPoseOpt).inverse();
            gtsam::Pose3 _2T1_ = T2.between(T1);
            sor.setInputCloud(akeyframe.KeyCloud);
            sor.filter(*cloud_ds_p);
            pcl::transformPointCloud(*(cloud_ds_p), *tmp_p, GTSPoseToEigenM4f(_2T1_));
        }

        PointType a_pt;
        a_pt.x = -1010.1; a_pt.y = i; a_pt.z = tmp_p->size();     // -1010.1 is the separation flag in stored pcd
        fullCorrected_p->push_back(a_pt);

        try
        {
            if (!tmp_p->empty())
                *fullCorrected_p += *(tmp_p);
        }catch(std::bad_alloc)
        {
            std::cerr << "std::bad_alloc" << std::endl ;
            fullCorrected_p->points.back().z = 0;
        }

        a_pt.x = akeyframe.KeyPoseOpt.roll; a_pt.y = akeyframe.KeyPoseOpt.pitch; a_pt.z = akeyframe.KeyPoseOpt.yaw;
        fullCorrected_p->push_back(a_pt);
        a_pt.x = akeyframe.KeyPoseOpt.x; a_pt.y = akeyframe.KeyPoseOpt.y; a_pt.z = akeyframe.KeyPoseOpt.z;
        opt_debug_file << "KF pose corrected: " << a_pt.x << " " << a_pt.y << " " << a_pt.z << endl;
        fullCorrected_p->push_back(a_pt);
        a_pt.x = akeyframe.KeyPose.roll; a_pt.y = akeyframe.KeyPose.pitch; a_pt.z = akeyframe.KeyPose.yaw;
        fullCorrected_p->push_back(a_pt);
        a_pt.x = akeyframe.KeyPose.x; a_pt.y = akeyframe.KeyPose.y; a_pt.z = akeyframe.KeyPose.z;
        opt_debug_file << "KF pose original: " << a_pt.x << " " << a_pt.y << " " << a_pt.z << endl;

        fullCorrected_p->push_back(a_pt);
        std::cout      << "[SavePrior]: prepare key frame: "  << i << std::endl;
        opt_debug_file << "[SavePrior]: prepare key frame: "  << i << std::endl;
    }
    gtsam::NonlinearFactorGraph graph_cur = isam_->getFactorsUnsafe();
    for (int i = 0; i < graph_cur.size(); i++)
    {
        if(factor_del.find(i) != factor_del.end())
            continue;
        PointType a_pt;
        auto afactor_keys = graph_cur[i]->keys();
        if (afactor_keys.size() == 2)
        {
            if (afactor_keys[1] - afactor_keys[0] > 2)
            {
                a_pt.x = -2020.2; a_pt.y = afactor_keys[0]; a_pt.z = afactor_keys[1];
            }
        }
        fullCorrected_p->push_back(a_pt);
    }

    if (!fullCorrected_p->empty())
    {
        std::string pcd_name = save_directory + "map_prior/clouds_corrected.pcd";
        opt_debug_file << "[SavePrior]: try saving prior map." << std::endl;
        pcl::io::savePCDFileBinary(pcd_name, *fullCorrected_p);
        opt_debug_file << "[SavePrior]: prior map saved as clouds_corrected.pcd " << std::endl;
    }
}

#ifdef as_node
int mainOptimizationFunction()
{
    int argc; char** argv;
    ros::init(argc, argv, "loop_optimization");
#else
int main(int argc, char **argv)
{
    ros::init(argc, argv, "loop_optimization");
#endif
    ROS_INFO("Loop Correction Node Starts");
    ros::NodeHandle nh;
    bool config_loaded = loadConfig(nh);
    if (!config_loaded) return 0;

    poses_opt_fname = save_directory + "optimized_poses.txt";

//    poses_opt_file = std::fstream(poses_opt_fname, std::fstream::out);
    pgo_scan_directory = save_directory + "scans";
    auto unused = system((std::string("mkdir -p ") + pgo_scan_directory).c_str());
    unused      = system((std::string("mkdir -p ") + pgo_scan_directory + "/FPR_Accepted").c_str());
    unused      = system((std::string("mkdir -p ") + pgo_scan_directory + "/FPR_Rejected").c_str());
    unused      = system((std::string("mkdir -p ") + pgo_scan_directory + "/LargeOverlap").c_str());
    unused      = system((std::string("mkdir -p ") + pgo_scan_directory + "/LowOverlap").c_str());
    unused      = system((std::string("mkdir -p ") + save_directory + "/map_prior").c_str());
    unused      = system((std::string("mkdir -p ") + save_directory + "/map_eva").c_str());

    poses_raw_fname = save_directory + "odom_poses.txt";
    poses_raw_file = std::fstream(poses_raw_fname, std::fstream::out);

//    times_file = std::fstream(save_directory + "timestamps.txt", std::fstream::out);
//    poses_opt_all_file.open(save_directory + "optimized_poses_full.txt", ios::out | ios::trunc);

    times_opt_file = std::fstream(save_directory + "times_optimization_LTAOM.txt", std::fstream::out);
    times_opt_file.precision(std::numeric_limits<double>::max_digits10);

    time_full_pgo_thread = std::fstream(save_directory + "times_full_loopoptimization_LTAOM.txt", std::fstream::out);
    time_full_pgo_thread.precision(std::numeric_limits<double>::max_digits10);

    lc_file = std::fstream(save_directory + "loopclosure.txt", std::fstream::out);
    opt_debug_file = std::fstream(save_directory + "opt_debug.txt", std::fstream::out);

    poses_opt_file.precision(std::numeric_limits<double>::max_digits10);
    poses_raw_file.precision(std::numeric_limits<double>::max_digits10);
//    poses_opt_all_file.precision(std::numeric_limits<double>::max_digits10);

    ros::Subscriber sub_cloud = nh.subscribe<sensor_msgs::PointCloud2>("/clouds_submap", 100, cloudCallback);
    ros::Subscriber sub_odom = nh.subscribe<nav_msgs::Odometry>("/submap_pose", 100, odometryCallback);
    ros::Subscriber sub_interT = nh.subscribe<sensor_msgs::PointCloud2>("/inter_triangles", 100, interKPPairCallback);
    ros::Subscriber sub_interT2 = nh.subscribe<sensor_msgs::PointCloud2>("/inter2_triangles", 100, inter2KPPairCallback);
    ros::Subscriber sub_lcT = nh.subscribe<sensor_msgs::PointCloud2>("/lc_triangles", 100, loopClosureKPPairCallback);
    ros::Subscriber sub_loopclosure = nh.subscribe<geometry_msgs::PoseWithCovariance>("/loop_closure_tranformation", 100, loopClosureCallback);
    ros::Subscriber sub_notification = nh.subscribe<std_msgs::Float64MultiArray>("/odom_correction_info64", 100, notificationCallback);
    ros::Subscriber sub_timeCorrection = nh.subscribe<std_msgs::UInt64>("/time_correction", 100, jumptimeCallback);

    pubOdomAftPGO = nh.advertise<nav_msgs::Odometry>("/aft_pgo_odom", 100);
    pubPathAftPGO = nh.advertise<nav_msgs::Path>("/aft_pgo_path", 100);
    pubPathAftPGOPrior = nh.advertise<nav_msgs::Path>("/aft_pgo_path_prior", 100);
    pubAndSaveGloablMapAftPGO = nh.advertise<sensor_msgs::PointCloud2>("/aft_pgo_map", 100);
    pubLoopClosure = nh.advertise<visualization_msgs::Marker>("/loopclosure", 100);

    parameters.relinearizeThreshold = 0.1;
    parameters.relinearizeSkip = 1;
    parameters.enableDetailedResults = true;
//    parameters.factorization = gtsam::ISAM2Params::QR; // This is setting make isam->update() robust but increase opt time cost
    isam_ = new gtsam::ISAM2(parameters);

    gtsam::Vector noise_vec6(6);
    noise_vec6 << 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12;
    pose_start_noise_ = gtsam::noiseModel::Diagonal::Variances(noise_vec6);
    noise_vec6 << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4;
    pose_noise_ = gtsam::noiseModel::Diagonal::Variances(noise_vec6);
if (multisession_mode == 1)
{
//    residual_thr = 5.0;
    loadPriorMapAndPoses();
}
    std::thread pose_graph_optimization {main_pgo};
    std::thread map_viz {processRvizMap};

    ros::Rate rate(5000);
    while(ros::ok())
    {
        rate.sleep();
        ros::spinOnce();
        bool save_one_time = false;
if (multisession_mode == 2)
{
        nh.getParam("/save_prior_info", save_one_time); // you can do [$rosparam set /save_prior_info true] on terminal to
                                                        // manually save prior info for future session
        if(save_one_time)
        {
            nh.setParam("/save_prior_info", false);
            save_one_time = false;
            mtx_keyf_read_.lock();
            savePrior();
            mtx_keyf_read_.unlock();
        }
}
        nh.getParam("/save_map", save_one_time);    // you can do [$rosparam set /save_map true] on terminal to manually save map
        if(save_one_time)
        {
            nh.setParam("/save_map", false);
            save_one_time = false;
            pubAndSaveGloablMap();
        }

        nh.getParam("/pub_pgopath", save_one_time);    // you can do [$rosparam set /pub_final_pgopath true] on terminal to manually save map
        if(save_one_time)
        {
            nh.setParam("/pub_pgopath", false);
            save_one_time = false;
            pubPath();
        }
    }

    pubAndSaveGloablMap();

    pose_graph_optimization.detach();
    map_viz.detach();
    poses_raw_file.close();
    poses_opt_file.close();
    lc_file.close();
    //poses_opt_all_file.close();
    times_opt_file.close();
    opt_debug_file.close();
    return 0;
}
