cmake_minimum_required(VERSION 3.8)
project(dressid)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
set(SOURCE_FILES main.cpp ANNtrainer.cpp ANNtrainer.hpp)
add_executable(dressid ${SOURCE_FILES})

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(dressid ${OpenCV_LIBS})