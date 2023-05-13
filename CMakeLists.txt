cmake_minimum_required(VERSION 3.17)
project(cudart-trace VERSION 1.0 DESCRIPTION "Trace cudart calls")

find_package(CUDAToolkit REQUIRED)
find_package(Boost)

add_library(cudart-trace SHARED
    cudart-trace.cpp
)

if(${Boost_FOUND})
    target_compile_definitions(cudart-trace PRIVATE HAVE_BOOST)
    target_include_directories(cudart-trace PRIVATE ${Boost_INCLUDE_DIRS})
    target_link_libraries(cudart-trace PRIVATE ${Boost_LIBRARIES})
endif()

target_include_directories(cudart-trace PRIVATE ${CUDAToolkit_INCLUDE_DIRS})