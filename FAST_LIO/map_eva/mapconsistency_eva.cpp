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
#include <math.h>
#include <fstream>
#include <unistd.h>
#include <so3_math.h>
#include <ros/ros.h>
#include <Eigen/Core>
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
#include <unordered_map>

#include <rosbag/bag.h>
#include <rosbag/view.h>
#define PI 3.14159265
//#define liosam
//std::ofstream outfile("/home/zuhaozou/ws_LTAOM/logs/map_eva.txt");

Eigen::Quaterniond EulerToEigenQuat(double roll, double pitch, double yaw){
  double c1 = cos(roll*0.5);
  double s1 = sin(roll*0.5);
  double c2 = cos(pitch*0.5);
  double s2 = sin(pitch*0.5);
  double c3 = cos(yaw*0.5);
  double s3 = sin(yaw*0.5);
  return Eigen::Quaterniond(c1*c2*c3 - s1*s2*s3, s1*c2*c3 + c1*s2*s3, -s1*c2*s3 + c1*s2*c3, c1*c2*s3 + s1*s2*c3);
}

struct imu_lidar_camera
{
    imu_lidar_camera()
    {
        msg_time = 0.0;
        lidar_poseset = 0;
        m_lidar_q = Eigen::Quaterniond::Identity();
        m_lidar_t = Eigen::Vector3d::Zero();
        pts_set = 0;
        ext_set = 0;
    };
    double msg_time;
    int lidar_poseset;
    int pts_set;
    int ext_set;
    Eigen::Quaterniond m_lidar_q;
    Eigen::Vector3d m_lidar_t;
    Eigen::Quaterniond m_ext_q;
    Eigen::Vector3d m_ext_t;
    pcl::PointCloud<pcl::PointXYZI> lidar_pts;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
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
    return ((((s.z) * HASH_P) % MAX_N + (s.y)) * HASH_P) % MAX_N + (s.x);
  }
};
} // namespace std

void SplitString(const std::string& s, std::vector<std::string>& v, const std::string& c)
{
     std::string::size_type pos1, pos2;
     pos2 = s.find(c);
     pos1 = 0;
     while(std::string::npos != pos2)
     {
         v.push_back(s.substr(pos1, pos2-pos1));

         pos1 = pos2 + c.size();
         pos2 = s.find(c, pos1);
     }
     if(pos1 != s.length())
         v.push_back(s.substr(pos1));
}
void loadPoses(std::map<std::string, imu_lidar_camera>& sorter, const std::string &file_name)
{
        int offset = 0;
        std::ifstream infile;
        infile.open(file_name);   //将文件流对象与文件连接起来
        assert(infile.is_open());   //若失败,则输出错误消息,并终止程序运行
        std::string s;
        while(getline(infile, s))
        {
            offset = 0;
            std::vector<std::string> tmp;
            SplitString(s, tmp, " ");
            std::string pose_time = tmp[offset+0];

            Eigen::Quaterniond w_q_l(atof(tmp[offset+7].c_str()), atof(tmp[offset+4].c_str()),
                            atof(tmp[offset+5].c_str()), atof(tmp[offset+6].c_str()));
            Eigen::Vector3d w_t_l(atof(tmp[offset+1].c_str()), atof(tmp[offset+2].c_str()),
                            atof(tmp[offset+3].c_str()));

            sorter[pose_time.erase(13)].m_lidar_q = w_q_l;
            sorter[pose_time.erase(13)].m_lidar_t = w_t_l;
            sorter[pose_time.erase(13)].lidar_poseset = 1;

            if (tmp.size() > 8)
            {
                offset = 7;
                Eigen::Quaterniond q_ext(atof(tmp[offset+7].c_str()), atof(tmp[offset+4].c_str()),
                    atof(tmp[offset+5].c_str()), atof(tmp[offset+6].c_str()));
                Eigen::Vector3d t_ext(atof(tmp[offset+1].c_str()), atof(tmp[offset+2].c_str()),
                    atof(tmp[offset+3].c_str()));
                sorter[pose_time.erase(13)].m_ext_q = q_ext;
                sorter[pose_time.erase(13)].m_ext_t = t_ext;
                sorter[pose_time.erase(13)].ext_set = 1;
            }

            //outfile << "loaded pose time " << pose_time.erase(13) << std::endl;
        }
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "map_eva");
    ros::NodeHandle nh;
    std::string bag_file = "";
    std::string pose_file = "";
    nh.param<std::string>("bag_file", bag_file, "");
    nh.param<std::string>("pose_file", pose_file, "");
    float voxel_size = 0.05f;
    ros::Publisher pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>
        ("/cloud_registered", 10000);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>
        ("/aft_mapped_to_init", 10000);

    Eigen::Vector3d extrinsic_T(1.77, 0.06, -0.05);
    Eigen::Matrix3d extrinsic_R;
    extrinsic_R << -1, 0, 0, // Mulran default extrinsic
                  0, -1, 0,
                  0, 0, 1;

    std::fstream file_;
    file_.open(bag_file, std::ios::in);
    if (!file_) {
      std::cout << "File " << bag_file << " does not exit" << std::endl;
    }
    ROS_INFO("Start to load the rosbag %s", bag_file.c_str());
    rosbag::Bag bag;
    try {
      bag.open(bag_file, rosbag::bagmode::Read);
    } catch (rosbag::BagException e) {
      ROS_ERROR_STREAM("LOADING BAG FAILED: " << e.what());
    }
    std::vector<std::string> types;
    types.push_back(std::string("sensor_msgs/PointCloud2"));
    rosbag::View view(bag, rosbag::TypeQuery(types));
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> scan_list;
    std::map<std::string, imu_lidar_camera> sorter;
    ROS_INFO("Start to load the pose %s", pose_file.c_str());
    loadPoses(sorter, pose_file);
#ifdef liosam
    ROS_INFO("Dealing with LIOSAM poses estimation");
#else
    ROS_INFO("Not dealing with LIOSAM poses estimation");
#endif

    std::unordered_map<VOXEL_LOC, int> voxel_map;
    int scan_cnt = 0;
    std::vector<double> msg_time_vec;
    ros::Rate rate(100);
    BOOST_FOREACH (rosbag::MessageInstance const m, view)
    {
      sensor_msgs::PointCloud2::ConstPtr cloud_ptr = m.instantiate<sensor_msgs::PointCloud2>();
      if (cloud_ptr != NULL)
      {
        double laser_time = cloud_ptr->header.stamp.toSec();
        //if (laser_time > 1566535887.249315)  continue;
        std::string laser_time_str = std::to_string(cloud_ptr->header.stamp.toSec()).erase(13);
        msg_time_vec.push_back(laser_time);
        pcl::PCLPointCloud2 pcl_pc;
        pcl::PointCloud<pcl::PointXYZI> undistorted_cloud;
        pcl_conversions::toPCL(*cloud_ptr, pcl_pc);
        pcl::fromPCLPointCloud2(pcl_pc, undistorted_cloud);

        sorter[laser_time_str].lidar_pts = undistorted_cloud;
        sorter[laser_time_str].pts_set = true;
        //outfile << laser_time_str << std::endl;
      }
    }

    for (int si = 0; si < msg_time_vec.size(); si++)
    {
        if (msg_time_vec[si] + 5 > msg_time_vec.back())
            continue;  // not benchmarking last 5 sec because other systems may no est pose when static (last 5 sec)
        std::string laser_time_str = std::to_string(msg_time_vec[si]).erase(13);
        Eigen::Vector3d translation;
        Eigen::Quaterniond q;
        //outfile << si << " " << laser_time_str << std::endl;
        if (sorter.find(laser_time_str) == sorter.end())
        {
            std::cout << "warn: sorter cannot find this laser cloud time" << std::endl;
            continue;
        }
        else
        {
            if (sorter[laser_time_str].pts_set)
            {
                if (sorter[laser_time_str].lidar_poseset)
                {
                  translation = sorter[laser_time_str].m_lidar_t;
                  q = Eigen::Quaterniond (sorter[laser_time_str].m_lidar_q);
                  if (sorter[laser_time_str].ext_set)
                  {
                    extrinsic_R = Eigen::Matrix3d(sorter[laser_time_str].m_ext_q);
                    extrinsic_T = sorter[laser_time_str].m_ext_t;
                  }
                }
                else  // if pose at the time not found, interpolate lidar's pose with surrounding est poses
                {
                  double pre_time = msg_time_vec[si];
                  bool pre_found = false;
                  Eigen::Vector3d pre_translation;
                  Eigen::Quaterniond pre_q;
                  while (pre_time - 0.01 > msg_time_vec[0])
                  {
                    pre_time = pre_time - 0.01;
                    std::string pre_time_str = std::to_string(pre_time).erase(13);
                    if (sorter[pre_time_str].lidar_poseset)
                    {
                      pre_translation = sorter[pre_time_str].m_lidar_t;
                      pre_q           = sorter[pre_time_str].m_lidar_q;
                      pre_found = true;
                      if (sorter[pre_time_str].ext_set)
                      {
                      extrinsic_R = Eigen::Matrix3d(sorter[pre_time_str].m_ext_q);
                      extrinsic_T = sorter[pre_time_str].m_ext_t;
                      }
                      break;
                    }
                  }
                  double nxt_time = msg_time_vec[si];
                  bool nxt_found = false;
                  Eigen::Vector3d nxt_translation;
                  Eigen::Quaterniond nxt_q;
                  while (nxt_time + 0.01 < msg_time_vec.back())
                  {
                    nxt_time = nxt_time + 0.01;
                    std::string nxt_time_str = std::to_string(nxt_time).erase(13);
                    if (sorter[nxt_time_str].lidar_poseset)
                    {
                      nxt_translation = sorter[nxt_time_str].m_lidar_t;
                      nxt_q           = sorter[nxt_time_str].m_lidar_q;
                      nxt_found = true;
                      if (sorter[nxt_time_str].ext_set)
                      {
                      extrinsic_R = Eigen::Matrix3d(sorter[nxt_time_str].m_ext_q);
                      extrinsic_T = sorter[nxt_time_str].m_ext_t;
                      }
                      break;
                    }
                  }

                  double dt_nxt = fabs(nxt_time - msg_time_vec[si]);
                  double dt_pre = fabs(pre_time - msg_time_vec[si]);
                  if (pre_found && !nxt_found)
                  {
                      translation = pre_translation ;
                      q = pre_q;
                  }
                  else if (nxt_found && !pre_found)
                  {
                      translation = nxt_translation ;
                      q = nxt_q;
                  }
                  else
                  {
                      translation = dt_nxt/(dt_nxt+dt_pre) *  pre_translation + dt_pre/(dt_nxt+dt_pre) *  nxt_translation ;
                      double t = (msg_time_vec[si] - pre_time) / (nxt_time - pre_time);
                      q = pre_q.slerp(t, nxt_q);
                  }

                  //outfile << "pre T: " << pre_translation.transpose() << std::endl;
                  //outfile << "nxt T: " << nxt_translation.transpose() << std::endl;
                }
                //outfile << "    T: " << translation.transpose() << std::endl;
                //outfile << "    q: " << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << " " << std::endl;
            }
            else
            {
              std::cout << "warn: lidar pt not set" << std::endl;
              continue;
            }
        }
        assert(sorter[laser_time_str].pts_set);
        pcl::PointCloud<pcl::PointXYZI> undistorted_cloud = sorter[laser_time_str].lidar_pts;
        nav_msgs::Odometry odom;
        odom.header.frame_id = "camera_init";
        odom.pose.pose.position.x = translation[0];
        odom.pose.pose.position.y = translation[1];
        odom.pose.pose.position.z = translation[2];
        odom.pose.pose.orientation.w = q.w();
        odom.pose.pose.orientation.x = q.x();
        odom.pose.pose.orientation.y = q.y();
        odom.pose.pose.orientation.z = q.z();
        pubOdomAftMapped.publish(odom);
        rate.sleep();

        pcl::PointCloud<pcl::PointXYZI>::Ptr register_cloud (new pcl::PointCloud<pcl::PointXYZI>());

        for (size_t i = 0; i < undistorted_cloud.size(); i++) {
          Eigen::Vector3d pv(undistorted_cloud.points[i].x, undistorted_cloud.points[i].y,
                             undistorted_cloud.points[i].z);
#ifdef liosam
          pv = q * EulerToEigenQuat(0,0,PI) * (extrinsic_R * pv + extrinsic_T) + translation; // liosam on mulran
#else
          pv = q * (extrinsic_R * pv + extrinsic_T) + translation;
#endif
          pcl::PointXYZI pi = undistorted_cloud.points[i];
          pi.x = pv[0];
          pi.y = pv[1];
          pi.z = pv[2];
          if (isnan(pi.x)||isnan(pi.y)||isnan(pi.z))  continue;
          register_cloud->push_back(pi);
        }

        sensor_msgs::PointCloud2Ptr laserCloudmsg (new sensor_msgs::PointCloud2());
        pcl::toROSMsg(*register_cloud, *laserCloudmsg);
        laserCloudmsg->header.stamp = ros::Time::now();
        laserCloudmsg->header.frame_id = "camera_init";
        pubLaserCloudFullRes.publish(laserCloudmsg);
        ros::spinOnce();
        rate.sleep();

        uint plsize = register_cloud->size();
        for (uint i = 0; i < plsize; i++)
        {
          pcl::PointXYZI &p_c = register_cloud->points[i];
          float loc_xyz[3];
          for (int j = 0; j < 3; j++) {
            loc_xyz[j] = p_c.data[j] / voxel_size;
            if (loc_xyz[j] < 0) {
              loc_xyz[j] -= 1.0;
            }
          }

          VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
              (int64_t)loc_xyz[2]);
          auto iter = voxel_map.find(position);
          if (iter != voxel_map.end())
            iter->second++;
          else
            voxel_map[position] = 1;
        }
        assert(si == scan_cnt);
        scan_cnt++;
        if (scan_cnt % 1000 == 0) std::cout << "scan_cnt: " << scan_cnt << ", voxel_map.size(): " << voxel_map.size() << std::endl;
        //outfile << "scan_cnt: " << scan_cnt << ", voxel_map.size(): " << voxel_map.size() << " " << laser_time_str << std::endl;
        //scan_list.push_back(register_cloud);

    }
    std::cout << "voxel_map.size(): " << voxel_map.size() << std::endl;
}
