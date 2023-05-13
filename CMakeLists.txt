cmake_minimum_required(VERSION 3.17)
project(cudart-trace VERSION 1.0 DESCRIPTION "Trace cudart calls")

find_package(CUDAToolkit REQUIRED)

add_library(cudart-trace SHARED
    cudart-trace.cpp
)

target_include_directories(cudart-trace PRIVATE ${CUDAToolkit_INCLUDE_DIRS})