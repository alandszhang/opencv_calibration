cmake_minimum_required(VERSION 3.16)
project(biaoding)

find_package(OpenCV)

set(CMAKE_CXX_STANDARD 14)
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(SOURCE_FILES main.cpp)
add_executable(biaoding ${SOURCE_FILES})


target_link_libraries(biaoding ${OpenCV_LIBS})