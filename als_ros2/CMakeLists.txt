cmake_minimum_required(VERSION 3.14)
project(als_ros2)

# ROS2 packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(pcl_conversions REQUIRED)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

add_executable(mcl src/mcl.cpp)
ament_target_dependencies(mcl
  rclcpp
  cv_bridge
  geometry_msgs
  sensor_msgs
  nav_msgs
  visualization_msgs
  tf2_ros
  tf2_geometry_msgs
)
target_link_libraries(mcl ${OpenCV_LIBRARIES} ${PCL_LIBRARIES} yaml-cpp)

add_executable(sm src/sm.cpp)
ament_target_dependencies(sm
  rclcpp
  cv_bridge
  geometry_msgs
  sensor_msgs
  nav_msgs
  tf2_ros
  tf2_geometry_msgs
  visualization_msgs
)
target_link_libraries(sm ${OpenCV_LIBRARIES} yaml-cpp)

add_executable(mrf_failure_detector src/mrf_failure_detector.cpp)
ament_target_dependencies(mrf_failure_detector
  rclcpp
  sensor_msgs
  geometry_msgs
  visualization_msgs
)
target_link_libraries(mrf_failure_detector yaml-cpp)

add_executable(gl_pose_sampler src/gl_pose_sampler.cpp)
ament_target_dependencies(gl_pose_sampler
  rclcpp
  tf2_ros
  tf2_geometry_msgs
  geometry_msgs
  sensor_msgs
  nav_msgs
  visualization_msgs
  cv_bridge
)
target_link_libraries(gl_pose_sampler ${OpenCV_LIBRARIES})

add_executable(classifier_dataset_generator src/classifier_dataset_generator.cpp)
ament_target_dependencies(classifier_dataset_generator
  rclcpp
  tf2_geometry_msgs
  sensor_msgs
  nav_msgs
  cv_bridge
)
target_link_libraries(classifier_dataset_generator ${OpenCV_LIBRARIES})

add_executable(mae_classifier_learning src/mae_classifier_learning.cpp)
ament_target_dependencies(mae_classifier_learning
  rclcpp
  tf2_geometry_msgs
  sensor_msgs
  nav_msgs
)
target_link_libraries(mae_classifier_learning ${OpenCV_LIBRARIES} yaml-cpp)

add_executable(evaluator src/evaluator.cpp)
ament_target_dependencies(evaluator
  rclcpp
  geometry_msgs
  sensor_msgs
  tf2_ros
  tf2_geometry_msgs
)
target_link_libraries(evaluator yaml-cpp)

add_executable(scan2pc src/scan2pc.cpp)
ament_target_dependencies(scan2pc
  rclcpp
  sensor_msgs
  geometry_msgs
)
target_link_libraries(scan2pc yaml-cpp)

add_executable(slamer src/slamer.cpp)
ament_target_dependencies(slamer
  rclcpp
  sensor_msgs
  geometry_msgs
  nav_msgs
  std_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  visualization_msgs
  cv_bridge
  pcl_conversions
)
target_link_libraries(slamer ${OpenCV_LIBRARIES} yaml-cpp ${PCL_LIBRARIES})

install(TARGETS
  mcl
  sm
  mrf_failure_detector
  gl_pose_sampler
  classifier_dataset_generator
  mae_classifier_learning
  evaluator
  scan2pc
  slamer
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  maps
  DESTINATION share/${PROJECT_NAME})

ament_package()