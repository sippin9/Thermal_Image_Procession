# CMakeList.txt: SVD 的 CMake 项目，在此处包括源代码并定义
# TODO: 如有需要，请添加测试并安装目标。
# CMakeLists.txt
cmake_minimum_required(VERSION 3.0)
project("SVD")

include_directories("F:/_SLAM/opencv/build/include")
link_directories("F:/_SLAM/opencv/build/x64/vc15/lib")

# Add executable
add_executable (SVD "main.cpp" "Tracking.cpp" "Tracking.h")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET SVD PROPERTY CXX_STANDARD 20)
endif()

# Link OpenCV library
target_link_libraries(SVD opencv_world3416.lib)
