cmake_minimum_required(VERSION 2.91)

project(levelset)

set(CMAKE_CXX_FLAGS "-std=c++11 -march=native -O3")

add_executable(${PROJECT_NAME} main.cpp pattern.cpp)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(OpenMP)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
