# LTAOM
## LTA-OM: Long-Term Association LiDAR-Inertial Odometry and Mapping [JFR2024]

## 1. Introduction
**LTA-OM** is an efficient, robust, and accurate SLAM system. It integrates LIO, loop detection, loop optimization, LTA modules. It also supports multi-session mode.
<div align="center">
<img src="https://github.com/hku-mars/LTAOM/blob/main/loop_optimization/description_images/LTAOM_pipeline.jpg" width = 98% />
</div>

### 1.1 Paper
Our paper is available on https://onlinelibrary.wiley.com/doi/10.1002/rob.22337 (open access).

### 1.2 Demo Video
Our demo video is available on https://youtu.be/DVwppEKlKps or https://www.bilibili.com/video/BV1rT42197Mg/?spm_id_from=333.999.0.0.

## 2. Tested Environment
### 2.0 **Ubuntu**
Our operation system is Ubuntu 18.

### 2.1 **ROS**
Following this [ROS Installation](http://wiki.ros.org/ROS/Installation) to install ROS and its additional pacakge. Our ROS version is Melodic.

### 2.2 **gtsam**
Our gtsam version is gtsam-4.0.3. Note that: you need to add some functions to original gtsam code before compiling, please see the note at the end of this Readme. 

### 2.3 **ceres**
Our ceres version is ceres-solver-1.14.0.

### 2.4 **PCL**
Our pcl version is pcl-1.9.

### 2.5 **gcc/g++**
Our gcc/g++ version is gcc/g++ -7.

### 2.6 **Eigen3**

### 2.7 **TBB**
Our tbb: Threading Building Blocks 2019 Update 9

Download it from github and build it in a folder. Link it using hard directory in STD Cmakelist.txt.

## 3. Build
Clone this repository and catkin_make:
```
cd ~/ws_LTAOM/src/
git clone https://github.com/hku-mars/LTAOM.git
cd ../
catkin_make
source ~/ws_LTAOM/devel/setup.bash
```

## 4. Run our examples
In : LiDAR & IMU messages (multisession mode: prior map, prior key poses, prior STD database)
Out: cloud_result.pcd & optimized_poses.txt

Please make sure the output directory in the launch file is set. Otherwise, you may see that the program is not running.

### 4.1 Mulran (Ouster)
Download dataset from https://sites.google.com/view/mulran-pr/download. 
Run
```
cd ~/ws_LTAOM/src/LTAOM
./run_nodelet_ouster.sh
./run_loop_optimization.sh

// When LTAOM execution is done, run the following to save corrected map
rosparam set /save_map true 
```

### 4.2 NCLT (Velodyne)
Download dataset from http://robots.engin.umich.edu/nclt/index.html#download. 
Run
```
cd ~/ws_LTAOM/src/LTAOM
./run_nodelet_velodyne.sh
./run_loop_optimization.sh

// When LTAOM execution is done, run the following to save corrected map
rosparam set /save_map true 
```

### 4.3 Multilevel (Livox avia)
Download dataset from my one drive: https://connecthkuhk-my.sharepoint.com/:f:/g/personal/u3008067_connect_hku_hk/EmhWsq8qR7ZAiVIf4joHetUBdQ-71rHTK1rZD9h2kBX5lg?e=JtDHdh (password: LTAOM-HKUMARS).
Run
```
cd ~/ws_LTAOM/src/LTAOM
./run_nodelet_avia.sh
./run_loop_optimization.sh

// When LTAOM execution is done, run the following to save corrected map
rosparam set /save_map true 
```

### 4.4 Multi-session mode
#### 4.4.1 first session
First open loop_optimization/run_all_mulran_multisession.launch, and set multisession_mode = 2 in launch file to allow you save LTAOM result.

Run
```
cd ~/ws_LTAOM/src/LTAOM
./run_nodelet_ouster_multisession.sh
./run_loop_optimization.sh
```
When you have executed LTAOM with one bag, you can save the LTAOM result for future session:
```
rosparam set /save_prior_info true
```
#### 4.4.1 second session
Then, set multisession_mode = 1 in launch file to allow you load previous LTAOM result. After that, you can run
```
cd ~/ws_LTAOM/src/LTAOM
./run_nodelet_mulran_multisession.sh
./run_loop_optimization.sh

// When LTAOM execution is done, run the following to save corrected map
rosparam set /save_map true 
```
with a new bag as the second session.


### 4.5 Generate dense poses for map evaluation
Uncomment the line "//#define save_for_mapconsistency_eva" in laserMapping.cpp. 

Execute LTAOM like step 4.1-4.3.

When loop optimization is done, you will get scanposes_corrected.txt. In addition, undistorted scans will be recorded as undistoted_scans.bag .

Note, when LTAOM execution is done, type the following command in the terminal to make sure that scanposes_corrected.txt is complete.
```
rosparam set /pub_pgopath true
```

Finally, follow FAST_LIO/map_eva/ReadMe.txt to do evaluation.

## 5. Example results

Nclt 20120202 map agianst statellite image
<div align="center">
<img src="https://github.com/hku-mars/LTAOM/blob/main/loop_optimization/description_images/nclt_20120202_statellite.png" width = 98% />
</div>

Comparison on Mulran Riverside02
<div align="center">
<img src="https://github.com/hku-mars/LTAOM/blob/main/loop_optimization/description_images/map_compare_riverside02.png" width = 98% />
</div>

Comparison on Mulran DCC02
<div align="center">
<img src="https://github.com/hku-mars/LTAOM/blob/main/loop_optimization/description_images/map_compare_DCC02.png" width = 98% />
</div>

Multilevel building (structurally-similar scene) map
<div align="center">
<img src="https://github.com/hku-mars/LTAOM/blob/main/loop_optimization/description_images/Multilevel_map.png" width = 98% />
</div>

Multisession map stitching reuslt
<div align="center">
<img src="https://github.com/hku-mars/LTAOM/blob/main/loop_optimization/description_images/multisession_map.png" width = 98% />
</div>

## 6. Report our problems and bugs
We know our packages might not completely stable at this stage, and we are working on improving the performance and reliability of our codes. So, if you have met any bug or problem, please feel free to draw an issue and I will respond ASAP.

## 7. Acknowledgments
In the development of LTAOM, we stand on the shoulders of the following repositories:
FASTLIO2, ikdtree, STD, gtsam.

## License
The source code is released under [GPLv2](http://www.gnu.org/licenses/) license.

If you use any code of this repo in your academic research, please cite **at least one** of our papers:
```
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
```

For commercial use, please contact me < zuhaozouATyahoo.com > and Dr. Fu Zhang < fuzhangAThku.hk >.

## TODO
- Loop optimization node cannot be correctly triggered as nodelet.
- Problem when saving large output pcd in loop optimization node.

## Note
// to the place between lines 97 ~ 99 of ISAM2.h

Values theta_bkq_;

VariableIndex variableIndex_bkq_;

mutable VectorValues delta_bkq_;

mutable VectorValues deltaNewton_bkq_;

mutable VectorValues RgProd_bkq_;

mutable KeySet deltaReplacedMask_bkq_;

NonlinearFactorGraph nonlinearFactors_bkq_;

mutable GaussianFactorGraph linearFactors_bkq_;

ISAM2Params params_bkq_;

mutable boost::optional&lt;double&gt; doglegDelta_bkq_;

KeySet fixedVariables_bkq_;

int update_count_bkq_;

------------------------------------------------------

// to the place between lines 97 ~ 99 of ISAM2.h

void backup(); // to line 199 of ISAM2.h

void recover(); // to line 200 of ISAM2.h

------------------------------------------------------

// from line 395 of ISAM2.cpp

void ISAM2::backup(){ 

  variableIndex_bkq_ = variableIndex_;

  theta_bkq_ = theta_;

  delta_bkq_ = delta_;

  deltaNewton_bkq_ = deltaNewton_;

  RgProd_bkq_ = RgProd_;

  nonlinearFactors_bkq_ = nonlinearFactors_;

  fixedVariables_bkq_ = fixedVariables_;

  update_count_bkq_ = update_count_;

  deltaReplacedMask_bkq_ = deltaReplacedMask_;

  linearFactors_bkq_ = linearFactors_;

  doglegDelta_bkq_ = doglegDelta_;

  params_bkq_ = params_;

  // nodes_bkq_ = nodes_;

  // roots_bkq_ = roots_;

}

void ISAM2::recover(){

  variableIndex_ = variableIndex_bkq_;

  theta_ = theta_bkq_;

  delta_ = delta_bkq_;

  deltaNewton_ = deltaNewton_bkq_;

  RgProd_ = RgProd_bkq_;

  nonlinearFactors_ = nonlinearFactors_bkq_;

  fixedVariables_ = fixedVariables_bkq_;

  update_count_ = update_count_bkq_;

  deltaReplacedMask_ = deltaReplacedMask_bkq_;

  linearFactors_ = linearFactors_bkq_;

  doglegDelta_ = doglegDelta_bkq_;

  params_ = params_bkq_;

  // nodes_ = nodes_bkq_;

  // roots_ = roots_bkq_;

}

// to line 427 of ISAM2.cpp
