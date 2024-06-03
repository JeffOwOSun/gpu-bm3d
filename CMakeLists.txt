cmake_minimum_required(VERSION 3.10)
project(GPU-BM3D)

# Set CXX compiler flags
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wno-unknown-pragmas -w -m64 -pthread")

# Add CUDA support
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARIES_DIRS})

# libpng
find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIR})

# Libraries
find_package(OpenCV REQUIRED)
list(APPEND LIBS ${OpenCV_LIBS})

# Source files
set(CUDA_SOURCES bm3d.cu bm3d.h)

# library for bm3d
CUDA_ADD_LIBRARY(bm3d SHARED
    ${CUDA_SOURCES}
  )
target_compile_features(bm3d PUBLIC cxx_std_11)
target_link_libraries(bm3d ${CUDA_LIBRARIES} pthread CUDA::cufft)

# excutables
add_executable(demo demo.cpp)
target_include_directories(demo PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(demo ${LIBS} X11 png bm3d)

add_executable(denoise denoise.cpp ${CUDA_SOURCES})
target_include_directories(denoise PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(denoise ${LIBS} X11 png bm3d)

# Executable for psnr
add_executable(psnr cal_psnr.cpp)
target_link_libraries(psnr png X11)
target_include_directories(psnr PRIVATE ${OpenCV_INCLUDE_DIRS} X11 pthread)

# Option to build Python bindings
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)

if(BUILD_PYTHON_BINDINGS)
    # pybind11
    find_package(pybind11 REQUIRED)
    include_directories(${pybind11_INCLUDE_DIRS})

    # Add the executable
    pybind11_add_module(pyGpuBM3D pybm3d.cpp)
    target_include_directories(pyGpuBM3D PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(pyGpuBM3D PRIVATE ${LIBS} bm3d)
endif()