cmake_minimum_required(VERSION 3.0.2)
project(std_loop)

## Compile as C++11, supported in ROS Kinetic and newer
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
SET(CMAKE_BUILD_TYPE "Release")
#SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
#SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  pcl_conversions
  pcl_ros
  roscpp
  rospy
  sensor_msgs
  std_msgs
  tf_conversions
  nodelet
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Threads)
find_package(Ceres REQUIRED)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

catkin_package(
 CATKIN_DEPENDS roscpp rospy std_msgs
)

set(LIB_OPENCV_IMGPROC_DIR /home/zuhaozou/tbb)
add_library(libtbb SHARED IMPORTED)
set_target_properties(libtbb PROPERTIES IMPORTED_LOCATION ${LIB_OPENCV_IMGPROC_DIR}/build/my_tbb_build_release/libtbb.so.2)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${LIB_OPENCV_IMGPROC_DIR}/include
)

#add_executable(std_loop_test src/main.cpp src/std.cpp )
#target_link_libraries(std_loop_test
#        ${catkin_LIBRARIES}
#        ${PCL_LIBRARIES}
#        ${OpenCV_LIBS}
#        ${CERES_LIBRARIES}
#        libtbb)

#add_executable(std_loop_detection src/bag_loop.cpp src/std.cpp )
#target_link_libraries(std_loop_detection
#        ${catkin_LIBRARIES}
#        ${PCL_LIBRARIES}
#        ${OpenCV_LIBS}
#        ${CERES_LIBRARIES}
#      libtbb)

#add_executable(loop_detection src/loop_detection.cpp src/std.cpp src/std_ba.cpp)
add_library(loop_detection src/loop_detection.cpp src/std.cpp src/std_ba.cpp)
target_link_libraries(loop_detection
        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}
        ${OpenCV_LIBS}
        ${CERES_LIBRARIES}
        libtbb
)

add_library(loop_detection_plugin
  src/loop_detection_plugin.cpp
)
target_link_libraries(loop_detection_plugin loop_detection ${catkin_LIBRARIES})
if(catkin_EXPORTED_LIBRARIES)
  add_dependencies(loop_detection_plugin ${catkin_EXPORTED_LIBRARIES})
endif()
