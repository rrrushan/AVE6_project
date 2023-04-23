# Install script for directory: /home/carla/carla-ros-bridge/catkin_ws/src/ros-bridge/carla_msgs

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/carla/carla-ros-bridge/catkin_ws/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/carla_msgs/srv" TYPE FILE FILES
    "/home/carla/carla-ros-bridge/catkin_ws/src/ros-bridge/carla_msgs/srv/SpawnObject.srv"
    "/home/carla/carla-ros-bridge/catkin_ws/src/ros-bridge/carla_msgs/srv/DestroyObject.srv"
    "/home/carla/carla-ros-bridge/catkin_ws/src/ros-bridge/carla_msgs/srv/GetBlueprints.srv"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/carla_msgs/msg" TYPE FILE FILES
    "/home/carla/carla-ros-bridge/catkin_ws/src/ros-bridge/carla_msgs/msg/CarlaBoundingBox.msg"
    "/home/carla/carla-ros-bridge/catkin_ws/src/ros-bridge/carla_msgs/msg/CarlaEgoVehicleControl.msg"
    "/home/carla/carla-ros-bridge/catkin_ws/src/ros-bridge/carla_msgs/msg/CarlaEgoVehicleStatus.msg"
    "/home/carla/carla-ros-bridge/catkin_ws/src/ros-bridge/carla_msgs/msg/CarlaEgoVehicleInfoWheel.msg"
    "/home/carla/carla-ros-bridge/catkin_ws/src/ros-bridge/carla_msgs/msg/CarlaEgoVehicleInfo.msg"
    "/home/carla/carla-ros-bridge/catkin_ws/src/ros-bridge/carla_msgs/msg/CarlaCollisionEvent.msg"
    "/home/carla/carla-ros-bridge/catkin_ws/src/ros-bridge/carla_msgs/msg/CarlaLaneInvasionEvent.msg"
    "/home/carla/carla-ros-bridge/catkin_ws/src/ros-bridge/carla_msgs/msg/CarlaWorldInfo.msg"
    "/home/carla/carla-ros-bridge/catkin_ws/src/ros-bridge/carla_msgs/msg/CarlaActorInfo.msg"
    "/home/carla/carla-ros-bridge/catkin_ws/src/ros-bridge/carla_msgs/msg/CarlaActorList.msg"
    "/home/carla/carla-ros-bridge/catkin_ws/src/ros-bridge/carla_msgs/msg/CarlaControl.msg"
    "/home/carla/carla-ros-bridge/catkin_ws/src/ros-bridge/carla_msgs/msg/CarlaStatus.msg"
    "/home/carla/carla-ros-bridge/catkin_ws/src/ros-bridge/carla_msgs/msg/CarlaTrafficLightInfo.msg"
    "/home/carla/carla-ros-bridge/catkin_ws/src/ros-bridge/carla_msgs/msg/CarlaTrafficLightInfoList.msg"
    "/home/carla/carla-ros-bridge/catkin_ws/src/ros-bridge/carla_msgs/msg/CarlaTrafficLightStatus.msg"
    "/home/carla/carla-ros-bridge/catkin_ws/src/ros-bridge/carla_msgs/msg/CarlaTrafficLightStatusList.msg"
    "/home/carla/carla-ros-bridge/catkin_ws/src/ros-bridge/carla_msgs/msg/CarlaWalkerControl.msg"
    "/home/carla/carla-ros-bridge/catkin_ws/src/ros-bridge/carla_msgs/msg/CarlaWeatherParameters.msg"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/carla_msgs/cmake" TYPE FILE FILES "/home/carla/carla-ros-bridge/catkin_ws/build/ros-bridge/carla_msgs/catkin_generated/installspace/carla_msgs-msg-paths.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/home/carla/carla-ros-bridge/catkin_ws/devel/include/carla_msgs")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/roseus/ros" TYPE DIRECTORY FILES "/home/carla/carla-ros-bridge/catkin_ws/devel/share/roseus/ros/carla_msgs")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/common-lisp/ros" TYPE DIRECTORY FILES "/home/carla/carla-ros-bridge/catkin_ws/devel/share/common-lisp/ros/carla_msgs")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/gennodejs/ros" TYPE DIRECTORY FILES "/home/carla/carla-ros-bridge/catkin_ws/devel/share/gennodejs/ros/carla_msgs")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  execute_process(COMMAND "/usr/bin/python3" -m compileall "/home/carla/carla-ros-bridge/catkin_ws/devel/lib/python3/dist-packages/carla_msgs")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python3/dist-packages" TYPE DIRECTORY FILES "/home/carla/carla-ros-bridge/catkin_ws/devel/lib/python3/dist-packages/carla_msgs")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/carla/carla-ros-bridge/catkin_ws/build/ros-bridge/carla_msgs/catkin_generated/installspace/carla_msgs.pc")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/carla_msgs/cmake" TYPE FILE FILES "/home/carla/carla-ros-bridge/catkin_ws/build/ros-bridge/carla_msgs/catkin_generated/installspace/carla_msgs-msg-extras.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/carla_msgs/cmake" TYPE FILE FILES
    "/home/carla/carla-ros-bridge/catkin_ws/build/ros-bridge/carla_msgs/catkin_generated/installspace/carla_msgsConfig.cmake"
    "/home/carla/carla-ros-bridge/catkin_ws/build/ros-bridge/carla_msgs/catkin_generated/installspace/carla_msgsConfig-version.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/carla_msgs" TYPE FILE FILES "/home/carla/carla-ros-bridge/catkin_ws/src/ros-bridge/carla_msgs/package.xml")
endif()

