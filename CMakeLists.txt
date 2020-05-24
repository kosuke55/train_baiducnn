cmake_minimum_required(VERSION 2.8.12)
project(feature_generator_pb)

find_package( Eigen3 REQUIRED )
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAG "${CMAKE_CXX_FLAGS} --std=c++11 -03")

include_directories(EIGEN3_INCLUDE_DIR)
include_directories(
  include
  )

add_subdirectory(pybind11)
pybind11_add_module(../scripts/create_dataset/feature_generator_pb src/feature_generator_pb.cpp)
