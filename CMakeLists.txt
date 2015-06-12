cmake_minimum_required(VERSION 2.8)

project(levelset)

set(CMAKE_CXX_FLAGS "-std=c++11 -march=native -O3")

find_package(OpenCV REQUIRED)
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
