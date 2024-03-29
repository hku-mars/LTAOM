To Evaluate LTA-OM: 

1. Uncomment "//#define save_for_mapconsistency_eva" defined in laserMapping.cpp and compile

2. $ ./run_nodelet_xxx.sh
   $ ./run_loop_optimization.sh

3. When ros bag plays end, type the following in a terminal:
   $ rosparam set /pub_pgopath true

4. Find ../logs/scanposes_corrected.txt ../logs/undistoted_scans.bag 

5. Open evaluate_map.sh. Specify bag_file and pose_file with the paths found at step 4.

6. $ source ~/ws_LTAOM/devel/setup.bash
   $ roslaunch fast_lio evaluate_map.sh

ps:
1. If you meet "[ERROR] LOADING BAG FAILED: Bag unindexed", $ rosbag reindex undistoted_scans.bag 
2. If you are evaluating LIOSAM-SC, Uncomment "//#define liosam" defined in mapconsistency_eva.cpp
3. Above step 3 is important step to make sure that scanposes_corrected.txt is complete.



