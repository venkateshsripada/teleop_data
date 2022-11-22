cmake_minimum_required(VERSION 3.0.2)
project(franka_teleoperation)

find_package(catkin REQUIRED COMPONENTS
  franka_control
  franka_description
  franka_example_controllers
  panda_moveit_config
  rospy

  controller_interface
  dynamic_reconfigure
  eigen_conversions
  franka_hw
  geometry_msgs
  franka_example_controllers
  hardware_interface
  tf
  tf_conversions
  message_generation
  pluginlib
  realtime_tools
  roscpp
)

find_package(Eigen3 REQUIRED)
find_package(Franka 0.8.0 REQUIRED)
   
generate_dynamic_reconfigure_options(
  cfg/panda_leader_compliance_param.cfg
)


catkin_package(
  INCLUDE_DIRS include 
  LIBRARIES  franka_teleoperation
  CATKIN_DEPENDS
    controller_interface
    dynamic_reconfigure
    eigen_conversions
    franka_hw
    geometry_msgs
    hardware_interface
    tf
    tf_conversions
    message_runtime
    franka_example_controllers
    pluginlib
    realtime_tools
    roscpp
  DEPENDS Franka
)
  

add_library(franka_teleoperation
  src/franka_leader_impedance_controller.cpp
  
)

add_dependencies(franka_teleoperation
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
  ${PROJECT_NAME}_gencfg  
  # ${PROJECT_NAME}_generate_messages_cpp
  # ${PROJECT_NAME}_gencpp
)
 
target_link_libraries(franka_teleoperation PUBLIC
  ${Franka_LIBRARIES}
  ${catkin_LIBRARIES}
)

target_include_directories(franka_teleoperation SYSTEM PUBLIC
  ${Franka_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)

target_include_directories(franka_teleoperation PUBLIC
  include
)




install(TARGETS franka_teleoperation
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)
install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
install(DIRECTORY config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
install(FILES franka_leader_impedance_controller_plugin.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

## Tools
include(${CMAKE_CURRENT_LIST_DIR}/../cmake/ClangTools.cmake OPTIONAL
  RESULT_VARIABLE CLANG_TOOLS
)
if(CLANG_TOOLS)
  file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
  file(GLOB_RECURSE HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
  )
  add_format_target(franka_teleoperation FILES ${SOURCES} ${HEADERS})
  add_tidy_target(franka_leader_controllers
    FILES ${SOURCES}
    DEPENDS franka_teleoperation
  )
endif() 
