cmake_minimum_required(VERSION 3.10)

project(visual_odom)

# set the project name
find_package( OpenCV REQUIRED)
find_package( NLopt REQUIRED)
find_package( ALGLIB REQUIRED PATHS home/alglib-cmake/build)

# add the executable
add_executable(visual_odom stereo_rectification.cpp)
#add_executable(visual_odom test.cpp)
target_link_libraries(visual_odom ${OpenCV_LIBS} ALGLIB)